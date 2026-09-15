"""Attack-path graph: predict how an attacker moves across a discovered network.

Produces a simplified, human-friendly topology with nodes (hosts) and edges
(potential pivots), and highlights the most likely path an attacker would take
through the network based on each host's exposure and sensitivity.

Nothing here generates real traffic — the graph is computed purely from the
host inventory returned by ``network_scanner.host_risk()`` and an optional
forecast summary.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SERVICE_RISK: Dict[str, int] = {
    "telnet": 5,
    "vnc": 5,
    "netbios-ssn": 4,
    "mssql": 4,
    "rdp": 4,
    "redis": 4,
    "elasticsearch": 4,
    "ssh": 3,
    "ftp": 3,
    "mysql": 3,
    "postgres": 3,
    "mongodb": 3,
}

_DB_PORTS = {3306, 5432, 1433, 27017, 6379, 9200}
_MGMT_PORTS = {22, 21, 23, 3389, 5900, 445, 139}
_RISK_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

# Layout x-coordinates per role
_X_EXTERNAL = 0.06
_X_EDGE = 0.32
_X_INTERNAL = 0.68
_X_SENSITIVE = 0.94


def _host_role(row: pd.Series) -> str:
    ports = _parse_ports(row.get("open_ports", ""))
    labels = [s.strip() for s in str(row.get("services", "")).split(",") if s.strip()]
    if ports & _DB_PORTS:
        return "database"
    if ports & _MGMT_PORTS:
        return "exposed"
    if row.get("risk_score", 0) >= 5:
        return "exposed"
    return "internal"


def _parse_ports(raw: str) -> set:
    return {int(p) for p in str(raw).split(",") if str(p).strip().isdigit()}


def _ports_overlap(a_ports: set, b_ports: set) -> bool:
    common = a_ports & b_ports
    return len(common) >= 1


def build_attack_path(
    hosts_df: pd.DataFrame,
    forecast_summary: Optional[Dict] = None,
) -> Dict:
    """Compute a simple attack-path graph from the host inventory.

    Returns a dict with ``nodes`` (list), ``edges`` (list), ``attack_path``
    (ordered node ids), and a human-readable ``summary``.
    """
    if hosts_df is None or hosts_df.empty:
        return {
            "nodes": [_attacker_node()],
            "edges": [],
            "attack_path": ["attacker"],
            "summary": {"description": "No hosts discovered — no attack path to analyse."},
        }

    nodes: List[Dict] = [_attacker_node()]
    edges: List[Dict] = []
    hosts_map: Dict[str, Dict] = {}   # ip -> node
    hosts_ports: Dict[str, set] = {}  # ip -> port set

    # Classify each discovered host
    roles: Dict[str, List[str]] = {"exposed": [], "database": [], "internal": []}
    for _, row in hosts_df.iterrows():
        ip = str(row["ip"])
        ports = _parse_ports(row.get("open_ports", ""))
        if not ports:
            continue
        role = _host_role(row)
        risk = str(row.get("risk_label", "LOW"))
        node = {
            "id": ip,
            "label": ip,
            "type": role,
            "risk": risk,
            "ports": str(row.get("open_ports", "")),
            "services": str(row.get("services", "")),
        }
        nodes.append(node)
        hosts_map[ip] = node
        hosts_ports[ip] = ports
        roles.setdefault(role, []).append(ip)

    # Build edges: attacker → exposed; exposed ↔ exposed; exposed → database
    for ip in roles["exposed"]:
        edges.append({
            "source": "attacker",
            "target": ip,
            "label": "entry point",
            "style": "dashed",
            "severity": hosts_map[ip]["risk"],
        })

    for a_ip in roles["exposed"]:
        for b_ip in roles["exposed"]:
            if a_ip >= b_ip:
                continue
            if _ports_overlap(hosts_ports[a_ip], hosts_ports[b_ip]):
                edges.append({
                    "source": a_ip,
                    "target": b_ip,
                    "label": "shared service",
                    "style": "solid",
                    "severity": "MEDIUM",
                })
        for db_ip in roles["database"]:
            edges.append({
                "source": a_ip,
                "target": db_ip,
                "label": "lateral move",
                "style": "dashed",
                "severity": hosts_map[db_ip]["risk"],
            })

    # Compute attack path
    entry = _best_entry(roles, hosts_map)
    target = _best_target(roles, hosts_map)
    path: List[str] = ["attacker"]
    if entry:
        path.append(entry)
        # pick intermediate exposed host that connects to target if not already entry
        if target and target != entry:
            path.append(target)

    return {
        "nodes": nodes,
        "edges": edges,
        "attack_path": path,
        "summary": _build_summary(path, hosts_map),
    }


def _attacker_node():
    return {"id": "attacker", "label": "Attacker", "type": "external",
            "risk": "CRITICAL", "ports": "", "services": ""}


def _best_entry(roles: Dict, hosts_map: Dict) -> Optional[str]:
    candidates = roles.get("exposed", [])
    if not candidates:
        return None
    return max(candidates, key=lambda ip: _RISK_RANK.get(hosts_map[ip]["risk"], 0))


def _best_target(roles: Dict, hosts_map: Dict) -> Optional[str]:
    candidates = roles.get("database", [])
    if not candidates:
        candidates = [ip for ip in roles.get("internal", []) if hosts_map.get(ip, {}).get("risk") in ("HIGH", "CRITICAL")]
    if not candidates:
        return roles.get("exposed", [None])[0]
    return max(candidates, key=lambda ip: _RISK_RANK.get(hosts_map[ip]["risk"], 0))


def _build_summary(path: List[str], hosts_map: Dict) -> Dict:
    if len(path) < 2:
        return {"description": "No exposed host was found — the network looks safe."}
    entry_ip = path[1]
    entry = hosts_map.get(entry_ip, {})
    entry_ports = entry.get("ports", "")
    entry_services = entry.get("services", "")

    if len(path) < 3:
        return {
            "entry_point": entry_ip,
            "target": entry_ip,
            "description": (
                f"Attacker enters via {entry_ip} ({entry_services or entry_ports}). "
                "No clear path beyond that host was found."
            ),
        }
    target_ip = path[2]
    target = hosts_map.get(target_ip, {})
    return {
        "entry_point": entry_ip,
        "target": target_ip,
        "entry_ports": entry_ports,
        "target_ports": target.get("ports", ""),
        "description": (
            f"Attacker enters via {entry_ip} ({entry_services or entry_ports}), "
            f"then pivots to the sensitive host at {target_ip} ({target.get('services', '')})."
        ),
    }
