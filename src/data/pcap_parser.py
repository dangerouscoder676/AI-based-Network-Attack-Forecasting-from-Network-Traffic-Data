import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from collections import defaultdict
from scapy.all import PcapReader, IP, TCP, UDP

class PcapParser:
    """
    Parses PCAP files into flow-level dataframes without running out of memory.
    """
    def __init__(self):
        self.flows: Dict[Tuple, Dict[str, Any]] = defaultdict(lambda: {
            'pkts': 0, 'bytes': 0, 'duration': 0.0,
            'start_time': 0.0, 'end_time': 0.0,
            'fwd_pkts': 0, 'bwd_pkts': 0,
            'fwd_bytes': 0, 'bwd_bytes': 0,
            'flags': {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0},
            'ttls': [], 'win_sizes': [], 'pkt_lens': [], 'iats': [],
            'last_time': 0.0
        })

    def _get_flow_key(self, pkt) -> Tuple[str, str, int, int, int]:
        """Generate a canonical 5-tuple for bidirectional flows."""
        if IP not in pkt:
            return None
            
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto
        
        src_port = 0
        dst_port = 0
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
            
        # Canonical order: smallest IP first
        if src_ip < dst_ip:
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)
            
    def _extract_flags(self, tcp_layer) -> Dict[str, bool]:
        """Extract TCP flags."""
        flags = tcp_layer.flags
        return {
            'SYN': bool(flags & 0x02),
            'ACK': bool(flags & 0x10),
            'FIN': bool(flags & 0x01),
            'RST': bool(flags & 0x04),
            'PSH': bool(flags & 0x08),
            'URG': bool(flags & 0x20)
        }

    def parse(self, pcap_path: str) -> pd.DataFrame:
        """
        Parse PCAP file and aggregate into flows.
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")
            
        try:
            with PcapReader(pcap_path) as pcap_reader:
                for pkt in pcap_reader:
                    try:
                        flow_key = self._get_flow_key(pkt)
                        if not flow_key:
                            continue
                            
                        # Extract features
                        timestamp = float(pkt.time)
                        pkt_len = len(pkt)
                        is_tcp = TCP in pkt
                        ttl = pkt[IP].ttl
                        
                        flow = self.flows[flow_key]
                        
                        # First packet in flow
                        if flow['pkts'] == 0:
                            flow['start_time'] = timestamp
                            flow['last_time'] = timestamp
                            
                        # Update IAT
                        if flow['pkts'] > 0:
                            flow['iats'].append(timestamp - flow['last_time'])
                        flow['last_time'] = timestamp
                        
                        # Basic aggregates
                        flow['pkts'] += 1
                        flow['bytes'] += pkt_len
                        flow['pkt_lens'].append(pkt_len)
                        flow['ttls'].append(ttl)
                        
                        # Directional counts
                        is_forward = (pkt[IP].src == flow_key[0])
                        if is_forward:
                            flow['fwd_pkts'] += 1
                            flow['fwd_bytes'] += pkt_len
                        else:
                            flow['bwd_pkts'] += 1
                            flow['bwd_bytes'] += pkt_len
                            
                        if is_tcp:
                            flow['win_sizes'].append(pkt[TCP].window)
                            flags = self._extract_flags(pkt[TCP])
                            for f, v in flags.items():
                                if v:
                                    flow['flags'][f] += 1
                                    
                    except Exception as e:
                        # Skip corrupted packets gracefully
                        continue
        except Exception as e:
            print(f"Error parsing PCAP: {e}")
            return pd.DataFrame()
            
        # Compile flows to dataframe
        flow_records = []
        for key, flow in self.flows.items():
            flow['end_time'] = flow['last_time']
            duration = flow['end_time'] - flow['start_time']
            flow['duration'] = duration
            
            iats = flow['iats']
            pkt_lens = flow['pkt_lens']
            ttls = flow['ttls']
            wins = flow['win_sizes']
            
            record = {
                'Src IP': key[0],
                'Dst IP': key[1],
                'Src Port': key[2],
                'Dst Port': key[3],
                'Protocol': key[4],
                'Timestamp': pd.to_datetime(flow['start_time'], unit='s'),
                'Flow Duration': duration * 1e6, # microsec typically
                'Tot Fwd Pkts': flow['fwd_pkts'],
                'Tot Bwd Pkts': flow['bwd_pkts'],
                'TotLen Fwd Pkts': flow['fwd_bytes'],
                'TotLen Bwd Pkts': flow['bwd_bytes'],
                'total_bytes': flow['bytes'],
                'total_packets': flow['pkts'],
                'SYN_Flag_Cnt': flow['flags']['SYN'],
                'ACK_Flag_Cnt': flow['flags']['ACK'],
                'FIN_Flag_Cnt': flow['flags']['FIN'],
                'RST_Flag_Cnt': flow['flags']['RST'],
                'PSH_Flag_Cnt': flow['flags']['PSH'],
                'URG_Flag_Cnt': flow['flags']['URG'],
            }
            
            # Additional statistics
            if iats:
                record['Flow IAT Mean'] = np.mean(iats) * 1e6
                record['Flow IAT Std'] = np.std(iats) * 1e6
                record['Flow IAT Max'] = np.max(iats) * 1e6
                record['Flow IAT Min'] = np.min(iats) * 1e6
            else:
                record['Flow IAT Mean'] = 0.0
                record['Flow IAT Std'] = 0.0
                record['Flow IAT Max'] = 0.0
                record['Flow IAT Min'] = 0.0
                
            if pkt_lens:
                record['Pkt Len Min'] = np.min(pkt_lens)
                record['Pkt Len Max'] = np.max(pkt_lens)
                record['Pkt Len Mean'] = np.mean(pkt_lens)
                record['Pkt Len Std'] = np.std(pkt_lens)
            else:
                record['Pkt Len Min'] = 0.0
                record['Pkt Len Max'] = 0.0
                record['Pkt Len Mean'] = 0.0
                record['Pkt Len Std'] = 0.0
                
            if ttls:
                record['TTL_Variance'] = np.var(ttls)
            else:
                record['TTL_Variance'] = 0.0
                
            if wins:
                record['Init Fwd Win Byts'] = wins[0] if flow['fwd_pkts'] > 0 else 0
            else:
                record['Init Fwd Win Byts'] = 0
                
            flow_records.append(record)
            
        return pd.DataFrame(flow_records)
