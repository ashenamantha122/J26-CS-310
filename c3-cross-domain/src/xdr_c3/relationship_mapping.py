"""Stage 3 - Cross-Domain Relationship Mapping: reusable pipeline code.

Used by the Stage 3 notebook (and imported by the Stage 3b notebook), so the
matching and linking rules are defined in one place only.

    from relationship_mapping import load_data, build_links_all
    host, iot = load_data(DATA_DIR, HOST_FILE, IOT_FILE)

Terminology
    host = host / process telemetry (one row = one host event)
    iot  = IoT network flows       (one row = one flow)
    node ids: "host:<row>" and "iot:<row>"  (row = 0-based position in the original file)

No ground-truth label column is read or used anywhere in this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FMT = "%d/%m/%Y %H:%M:%S"
KEYS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
NET_EVENTS = ["network_connect", "network_accept"]

W_MATCH = 30            # seconds, C1<->C2 correlation window (selected in the notebook, Step 4)
HOST_HOST_TOL = 5       # seconds, connect <-> accept on two hosts
FLOW_GAP = 15 * 60      # seconds, flow <-> flow continuity
ENTRY_GAP = 15 * 60     # seconds, accept -> child process
FALLBACK_GAP = 5        # seconds, accept -> child whose parent pid has no row
SERVICE_MIN_EVENTS = 30 # a pid with this many events (or PID 1) is a long-lived service


# --------------------------------------------------------------------------- loading
def load_data(data_dir, host_file, iot_file):
    """Read both files as text, parse timestamps strictly, add row ids."""
    host = pd.read_csv(Path(data_dir) / host_file, dtype=str, keep_default_na=False)
    iot = pd.read_csv(Path(data_dir) / iot_file, dtype=str, keep_default_na=False)
    host["ts"] = pd.to_datetime(host["timestamp"], format=FMT)
    iot["ts"] = pd.to_datetime(iot["timestamp"], format=FMT)
    host["row"] = np.arange(len(host))
    iot["row"] = np.arange(len(iot))
    for c in ["process_id", "parent_process_id"]:
        host[c] = host[c].astype("int64")
    return host, iot


def time_slice(df, start, end):
    return df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))]


# --------------------------------------------------------------------------- Section 1: IoT <-> host
def network_events(host):
    """Host connect/accept events in which the host itself is one endpoint."""
    net = host[host["event_type"].isin(NET_EVENTS)]
    return net[(net["src_ip"] == net["host_ip"]) | (net["dst_ip"] == net["host_ip"])]


def relevant_flows(iot, managed_ips):
    """IoT flows with a managed host at one end (the only ones that can match a host event)."""
    return iot[iot["src_ip"].isin(managed_ips) | iot["dst_ip"].isin(managed_ips)]


def candidate_pairs(rel, net):
    """All (flow, host event) pairs with the same IP pair, ports and protocol,
    in the direct form and in the reversed form (source/destination swapped)."""
    cols = KEYS + ["row", "ts", "event_type", "process_name", "host_ip", "process_id"]
    direct = rel.merge(net[cols], on=KEYS, suffixes=("", "_host"))
    direct["orient"] = "direct"
    swapped = net[cols].rename(columns={"src_ip": "dst_ip", "dst_ip": "src_ip",
                                        "src_port": "dst_port", "dst_port": "src_port"})
    reverse = rel.merge(swapped, on=KEYS, suffixes=("", "_host"))
    reverse["orient"] = "reversed"
    cand = pd.concat([direct, reverse], ignore_index=True).rename(
        columns={"row": "iot_row", "row_host": "host_row", "ts": "iot_ts", "ts_host": "host_ts"})
    cand["dt_sec"] = (cand["iot_ts"] - cand["host_ts"]).abs().dt.total_seconds()
    return cand


def one_to_one(df, a="iot_row", b="host_row", gap="dt_sec"):
    """Nearest timestamp wins; every item on either side is used at most once."""
    df = df.sort_values(gap, kind="stable")
    return df.drop_duplicates(a).drop_duplicates(b)


def match_iot_host(host, iot, managed_ips, window=W_MATCH):
    cand = candidate_pairs(relevant_flows(iot, managed_ips), network_events(host))
    return one_to_one(cand[cand["dt_sec"] <= window]).sort_values("iot_ts")


# --------------------------------------------------------------------------- Section 2: other links
def _node(prefix, rows):
    return prefix + ":" + pd.Series(rows).astype(str).to_numpy()


def _links(a_prefix, a_rows, b_prefix, b_rows, link_type, points, gap):
    return pd.DataFrame({"node_a": _node(a_prefix, a_rows), "node_b": _node(b_prefix, b_rows),
                         "link_type": link_type, "points": points, "gap_sec": np.asarray(gap, dtype=float)})


def host_host_links(host, managed_ips, tol=HOST_HOST_TOL):
    """connect on Host A  <->  accept on Host B, same tuple, within `tol` seconds."""
    net = network_events(host)
    cols = KEYS + ["row", "ts", "host_ip"]
    conn = net[(net["event_type"] == "network_connect") & (net["src_ip"] == net["host_ip"]) &
               net["dst_ip"].isin(managed_ips)][cols]
    acc = net[(net["event_type"] == "network_accept") & net["src_ip"].isin(managed_ips)][cols]
    m = conn.merge(acc, on=KEYS, suffixes=("_conn", "_acc"))
    m = m[m["host_ip_conn"] != m["host_ip_acc"]]
    m["gap"] = (m["ts_conn"] - m["ts_acc"]).abs().dt.total_seconds()
    m = m[m["gap"] <= tol].sort_values("gap", kind="stable").drop_duplicates("row_conn").drop_duplicates("row_acc")
    return _links("host", m["row_conn"], "host", m["row_acc"], "connect_accept", 3, m["gap"])


def flow_links(iot, gap=FLOW_GAP):
    """Consecutive flows that share a source and/or a destination within `gap` seconds.
    Same source AND destination is the strong link; source-only / destination-only are weak."""
    f = iot.sort_values(["ts", "row"], kind="stable")[["row", "ts", "src_ip", "dst_ip"]].reset_index(drop=True)

    def consecutive(keys):
        g = f.groupby(keys, sort=False)
        prev = g["row"].shift()
        d = (f["ts"] - g["ts"].shift()).dt.total_seconds()
        ok = prev.notna() & (d <= gap)
        return pd.DataFrame({"prev": prev[ok].astype(int), "cur": f.loc[ok, "row"], "gap": d[ok]})

    both = consecutive(["src_ip", "dst_ip"])
    seen = set(zip(both["prev"], both["cur"]))
    parts = [_links("iot", both["prev"], "iot", both["cur"], "flow_same_src_dst", 2, both["gap"])]
    for keys, ltype in ((["src_ip"], "flow_same_src"), (["dst_ip"], "flow_same_dst")):
        c = consecutive(keys)
        keep = np.array([(p, r) not in seen for p, r in zip(c["prev"], c["cur"])], dtype=bool)
        c = c[keep]
        parts.append(_links("iot", c["prev"], "iot", c["cur"], ltype, 1, c["gap"]))
    return pd.concat(parts, ignore_index=True)


def service_pids(host, min_events=SERVICE_MIN_EVENTS):
    """Long-lived services (many events) and PID 1: a shared service is NOT a shared entity."""
    counts = host["process_id"].value_counts()
    return set(counts[counts >= min_events].index) | {1}


def process_links(host, services):
    """Process-lineage links. Every link joins an earlier event to a later one."""
    h = host.sort_values(["ts", "row"], kind="stable").reset_index(drop=True)
    all_pids = set(h["process_id"])
    ps = h[h["event_type"] == "process_start"]
    acc = h[h["event_type"] == "network_accept"]
    links = []

    # (a) consecutive events of the same ephemeral (non-service) process
    eph = h[~h["process_id"].isin(services)]
    g = eph.groupby(["host_ip", "process_id"], sort=False)
    prev_row, prev_ts = g["row"].shift(), g["ts"].shift()
    d = (eph["ts"] - prev_ts).dt.total_seconds()
    ok = prev_row.notna() & (d <= ENTRY_GAP)
    links.append(_links("host", prev_row[ok].astype(int), "host", eph["row"][ok], "same_process", 2, d[ok]))

    def asof(left, right, key_left, key_right, by_left, tol):
        l = left[["row", "ts", "host_ip", key_left]].rename(columns={"row": "child_row", "ts": "child_ts", key_left: "_k"})
        r = right[["row", "ts", "host_ip", key_right]].rename(columns={"row": "prev_row", "ts": "prev_ts", key_right: "_k"})
        l = l.sort_values("child_ts", kind="stable")
        r = r.sort_values("prev_ts", kind="stable")
        by = ["host_ip", "_k"] if by_left else ["host_ip"]
        m = pd.merge_asof(l, r, left_on="child_ts", right_on="prev_ts", by=by,
                          direction="backward", tolerance=pd.Timedelta(seconds=tol))
        m = m.dropna(subset=["prev_row"])
        m["gap"] = (m["child_ts"] - m["prev_ts"]).dt.total_seconds()
        return m

    # (b) child process_start <-> latest event of its (non-service) parent
    ps_ns = ps[~ps["parent_process_id"].isin(services) & ps["parent_process_id"].isin(all_pids)]
    m = asof(ps_ns, h, "parent_process_id", "process_id", True, ENTRY_GAP)
    links.append(_links("host", m["prev_row"].astype(int), "host", m["child_row"], "parent_child", 2, m["gap"]))

    # (c) entry-to-child: child of a SERVICE process <-> nearest preceding accept handled by that service
    ps_sv = ps[ps["parent_process_id"].isin(services) & (ps["parent_process_id"] != 1)]
    m = asof(ps_sv, acc, "parent_process_id", "process_id", True, ENTRY_GAP)
    pts = np.where(m["gap"] <= 5, 3, 2)
    links.append(_links("host", m["prev_row"].astype(int), "host", m["child_row"], "entry_to_child", pts, m["gap"]))

    # (d) fallback: parent pid never appears as a row -> nearest preceding accept on the same host
    ps_or = ps[~ps["parent_process_id"].isin(all_pids)]
    m = asof(ps_or, acc, "parent_process_id", "process_id", False, FALLBACK_GAP)
    links.append(_links("host", m["prev_row"].astype(int), "host", m["child_row"], "fallback_accept", 2, m["gap"]))
    return pd.concat(links, ignore_index=True)


def build_links_all(host, iot, managed_ips, window=W_MATCH):
    """Every link used by the scenario grouping (Stage 3b): one row per link."""
    m = match_iot_host(host, iot, managed_ips, window)
    iot_host = _links("iot", m["iot_row"], "host", m["host_row"], "iot_host_match", 3, m["dt_sec"])
    parts = [iot_host, host_host_links(host, managed_ips), flow_links(iot),
             process_links(host, service_pids(host))]
    links = pd.concat(parts, ignore_index=True)
    return links.drop_duplicates(["node_a", "node_b", "link_type"]).reset_index(drop=True)


# =========================================================================== application integration
# Everything above this line is the Stage 3 notebook module, extracted verbatim (cell 5 of
# Stage3_Cross_Domain_Relationship_Mapping_V6v2.ipynb). Only the adapter below is new.
LINK_COLUMNS = ["node_a", "node_b", "link_type", "points", "gap_sec"]


def build_links_from_stage2(data, managed_ips=None, window=W_MATCH):
    """Run the original Stage 3 link builder on validated Stage 2 data.

    `data` is an `xdr_c3.stage2_validate.Stage2Data`. Its inference frames already hold `ts` and the 0-based
    original `row` ids, so no timestamp is parsed a second time, and they contain no ground-truth columns
    (label / type / attack_type), so relationships cannot depend on them. `managed_ips` defaults to the host IPs
    of the supplied host frame, as in the notebook (taken from the FULL host file).
    Returns the same table as `build_links_all` (columns: node_a, node_b, link_type, points, gap_sec).
    """
    from .stage2_validate import assert_no_ground_truth

    host, iot = data.to_stage3_frames(include_ground_truth=False)
    assert_no_ground_truth(host)
    assert_no_ground_truth(iot)
    if managed_ips is None:
        managed_ips = set(host["host_ip"])
    return build_links_all(host, iot, managed_ips, window)
