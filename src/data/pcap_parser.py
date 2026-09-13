import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from collections import defaultdict
from scapy.all import PcapReader, IP, TCP, UDP

class FlowAggregator:
    """
    Incremental bidirectional flow aggregation from raw packets.

    Exposes the same statistics as the offline [:class:`PcapParser`] but can be
    fed packet-by-packet, which makes it usable for live capture and PCAP replay.
    Thread safety is left to the caller (see ``live_capture.LiveTrafficEngine``).
    """

    _FLOW_TEMPLATE = {
        'pkts': 0, 'bytes': 0, 'duration': 0.0,
        'start_time': 0.0, 'end_time': 0.0,
        'fwd_pkts': 0, 'bwd_pkts': 0,
        'fwd_bytes': 0, 'bwd_bytes': 0,
        'flags': {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0},
        'ttls': [], 'win_sizes': [], 'pkt_lens': [], 'iats': [],
        'last_time': 0.0
    }

    def __init__(self):
        self.flows: Dict[Tuple, Dict[str, Any]] = defaultdict(dict)
        self._reset_flow(self.flows)

    @classmethod
    def _reset_flow(cls, flows: Dict[Tuple, Dict[str, Any]]):
        pass

    def get_flow(self, key: Tuple):
        flow = self.flows.get(key)
        if flow is None:
            flow = {k: (dict(v) if isinstance(v, dict) else ([] if isinstance(v, list) else v))
                    for k, v in self._FLOW_TEMPLATE.items()}
            self.flows[key] = flow
        return flow

    def reset(self):
        self.flows.clear()

    def _get_flow_key(self, pkt) -> Tuple[str, str, int, int, int]:
        """Generate a canonical 5-tuple for bidirectional flows."""
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
        elif 'IP' in pkt:
            src_ip = pkt['IP'].src
            dst_ip = pkt['IP'].dst
            proto = pkt['IP'].proto
        else:
            return None

        src_port = 0
        dst_port = 0
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        elif 'TCP' in pkt:
            src_port = pkt['TCP'].sport
            dst_port = pkt['TCP'].dport
        elif 'UDP' in pkt:
            src_port = pkt['UDP'].sport
            dst_port = pkt['UDP'].dport

        # Canonical order: smallest IP first
        if src_ip < dst_ip:
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)

    def _extract_flags(self, tcp_layer) -> Dict[str, bool]:
        """Extract TCP flags."""
        try:
            flags = tcp_layer.flags
        except AttributeError:
            return {}
        return {
            'SYN': bool(flags & 0x02),
            'ACK': bool(flags & 0x10),
            'FIN': bool(flags & 0x01),
            'RST': bool(flags & 0x04),
            'PSH': bool(flags & 0x08),
            'URG': bool(flags & 0x20)
        }

    def update(self, pkt, timestamp: Any = None) -> bool:
        """
        Feed a single packet into the aggregator.

        ``timestamp`` overrides the packet's own capture time (used to project
        PCAP replay timestamps onto a compressed demo clock). Returns False for
        packets that carry no usable flow (non-IP, corrupted).
        """
        if pkts := getattr(pkt, 'haslayer', None):
            if not pkt.haslayer('IP'):
                return False
        try:
            flow_key = self._get_flow_key(pkt)
            if not flow_key:
                return False

            if timestamp is None:
                timestamp = float(pkt.time)

            pkt_len = len(pkt)
            is_tcp = TCP in pkt or pkt.haslayer('TCP')
            ttl = pkt[IP].ttl if IP in pkt else pkt['IP'].ttl

            flow = self.get_flow(flow_key)

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
            src_ip = pkt[IP].src if IP in pkt else pkt['IP'].src
            is_forward = (src_ip == flow_key[0])
            if is_forward:
                flow['fwd_pkts'] += 1
                flow['fwd_bytes'] += pkt_len
            else:
                flow['bwd_pkts'] += 1
                flow['bwd_bytes'] += pkt_len

            if is_tcp:
                tcp_layer = pkt[TCP] if TCP in pkt else pkt['TCP']
                try:
                    flow['win_sizes'].append(int(tcp_layer.window))
                except (AttributeError, ValueError):
                    pass
                flags = self._extract_flags(tcp_layer)
                for f, v in flags.items():
                    if v:
                        flow['flags'][f] += 1
            return True
        except Exception:
            # Skip corrupted / unsupported packets gracefully
            return False

    def snapshot(self) -> pd.DataFrame:
        """Compile the current flows into a flow-level DataFrame."""
        flow_records = []
        for key, flow in self.flows.items():
            flow['end_time'] = flow['last_time']
            duration = flow['end_time'] - flow['start_time']
            flow['duration'] = duration

            iats = flow['iats']
            pkt_lens = flow['pkt_lens']
            ttls = flow['ttls']
            wins = flow['win_sizes']
            fwd_pkts = max(flow['fwd_pkts'], 1e-9)

            record = {
                'Src IP': key[0],
                'Dst IP': key[1],
                'Src Port': key[2],
                'Dst Port': key[3],
                'Protocol': key[4],
                'Timestamp': pd.to_datetime(flow['start_time'], unit='s'),
                'Flow Duration': duration * 1e6,  # microsec typical
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
                record['Init Fwd Win Byts'] = wins[0] if fwd_pkts > 0 else 0
            else:
                record['Init Fwd Win Byts'] = 0

            flow_records.append(record)

        return pd.DataFrame(flow_records)


class PcapParser:
    """
    Parses PCAP files into flow-level dataframes without running out of memory.
    Streaming (live) consumers use the underlying :class:`FlowAggregator`.
    """
    def __init__(self, aggregator: FlowAggregator = None):
        self._aggregator = aggregator or FlowAggregator()

    @property
    def flows(self) -> Dict[Tuple, Dict[str, Any]]:
        return self._aggregator.flows

    def update(self, pkt, timestamp=None):
        """Stream a single packet into the aggregator (live mode)."""
        return self._aggregator.update(pkt, timestamp=timestamp)

    def snapshot(self) -> pd.DataFrame:
        """Return the current aggregated flows (live mode)."""
        return self._aggregator.snapshot()

    def _get_flow_key(self, pkt) -> Tuple[str, str, int, int, int]:
        """Generate a canonical 5-tuple for bidirectional flows."""
        return self._aggregator._get_flow_key(pkt)

    def _extract_flags(self, tcp_layer) -> Dict[str, bool]:
        """Extract TCP flags."""
        return self._aggregator._extract_flags(tcp_layer)

    def parse(self, pcap_path: str) -> pd.DataFrame:
        """
        Parse PCAP file and aggregate into flows.
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP not found: {pcap_path}")

        self._aggregator.reset()
        try:
            with PcapReader(pcap_path) as pcap_reader:
                for pkt in pcap_reader:
                    self._aggregator.update(pkt)
        except Exception as e:
            print(f"Error parsing PCAP: {e}")
            return pd.DataFrame()

        return self._aggregator.snapshot()
