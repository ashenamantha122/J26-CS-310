"""Stage 3b - Scenario-Based Grouping: reusable pipeline code.

Takes the links from Stage 3 (relationship_mapping.py) and turns them into
scenarios (groups of events that belong to the same story).

    import relationship_mapping as rm, scenario_grouping as sg
    nodes = sg.node_table(host, iot, managed_ips)
    pairs = sg.pair_scores(links_all, nodes, len(host))
    scenario = sg.group_scenarios(pairs, len(nodes), nodes["ts_s"].to_numpy(), T=3)

Grouping uses ONLY structure (time, addresses, process lineage and the Stage 3 links).
The ground-truth label is read only by the audit / labelling functions at the bottom,
which are used for validation, and the normal / attack-containing naming.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

GATE_GAP = 15 * 60       # seconds: two events further apart than this are never in one scenario
SHORT_GAP = 5 * 60       # seconds: "short gap" bonus point


# --------------------------------------------------------------------------- nodes and pairs
def node_table(host, iot, managed_ips):
    """One row per node. Index: host rows first (0..n_host-1), then iot rows (n_host..)."""
    n_host = len(host)
    iot_host_ip = np.where(iot["src_ip"].isin(managed_ips), iot["src_ip"],
                           np.where(iot["dst_ip"].isin(managed_ips), iot["dst_ip"], ""))
    nodes = pd.DataFrame({
        "node": np.concatenate(["host:" + host["row"].astype(str).to_numpy(), "iot:" + iot["row"].astype(str).to_numpy()]),
        "kind": ["host"] * n_host + ["iot"] * len(iot),
        "ts_s": np.concatenate([host["ts"].astype("datetime64[s]").astype("int64").to_numpy(),
                                iot["ts"].astype("datetime64[s]").astype("int64").to_numpy()]).astype(np.int64),
        "host_ip": np.concatenate([host["host_ip"].to_numpy(object), iot_host_ip.astype(object)]),
        "label": np.concatenate([host["label"].astype(int).to_numpy(), iot["label"].astype(int).to_numpy()]),
    })
    return nodes


def node_index(ids, n_host):
    kind = ids.str.split(":").str[0].to_numpy()
    row = ids.str.split(":").str[1].astype(int).to_numpy()
    return np.where(kind == "host", row, n_host + row)


def pair_scores(links, nodes, n_host):
    """Add up the evidence for every linked pair of events and apply the time gate.
    score = points of all links between the two events + 1 (same host) + 1 (gap under 5 min)."""
    df = pd.DataFrame({"a": node_index(links["node_a"], n_host), "b": node_index(links["node_b"], n_host),
                       "points": links["points"].to_numpy()})
    pair = df.groupby(["a", "b"], as_index=False)["points"].sum()
    ts, hip = nodes["ts_s"].to_numpy(), nodes["host_ip"].to_numpy(object)
    pair["gap"] = np.abs(ts[pair["b"]] - ts[pair["a"]])
    ha, hb = hip[pair["a"]], hip[pair["b"]]
    pair["same_host"] = ((ha == hb) & (ha != "")).astype(int)
    pair["short_gap"] = (pair["gap"] < SHORT_GAP).astype(int)
    pair["score"] = pair["points"] + pair["same_host"] + pair["short_gap"]
    return pair[pair["gap"] <= GATE_GAP].reset_index(drop=True)


# --------------------------------------------------------------------------- grouping
def group_scenarios(pairs, n_nodes, ts_s, T):
    """Link pairs with score >= T, take connected components, then split a component wherever
    two consecutive events are more than 15 minutes apart. Returns one scenario id per node."""
    acc = pairs[pairs["score"] >= T]
    graph = coo_matrix((np.ones(len(acc), dtype=np.int8), (acc["a"].to_numpy(), acc["b"].to_numpy())), shape=(n_nodes, n_nodes))
    _, comp = connected_components(graph, directed=False)
    order = np.lexsort((ts_s, comp))
    c, t = comp[order], ts_s[order]
    new_piece = np.ones(len(order), dtype=bool)
    new_piece[1:] = (c[1:] != c[:-1]) | ((t[1:] - t[:-1]) > GATE_GAP)
    sid = np.empty(n_nodes, dtype=np.int64)
    sid[order] = np.cumsum(new_piece) - 1
    return sid


def sweep_row(scenario, T):
    size = np.bincount(scenario)
    return {"T": T, "scenarios": len(size), "scenarios_2plus": int((size >= 2).sum()),
            "events_in_2plus": int(size[size >= 2].sum()), "singleton_share_%": round((size == 1).mean() * 100, 1),
            "mean_size_2plus": round(float(size[size >= 2].mean()), 2), "max_size": int(size.max()),
            "scenarios_over_200": int((size > 200).sum())}


# --------------------------------------------------------------------------- audits (structure only unless noted)
def recover_chains(host, iot, links, n_host):
    """IoT flow -> Host A accept -> (descendant process) connect -> Host B accept, found from exported fields only.
    Returns the four node indices of every chain."""
    ppid_of = dict(zip(host["process_id"].tolist(), host["parent_process_id"].tolist()))

    def lineage(pid):
        out, cur = {pid}, pid
        for _ in range(12):
            nxt = ppid_of.get(cur)
            if nxt is None or nxt == 1 or nxt in out:
                break
            out.add(nxt)
            cur = nxt
        return out

    ih = links[links["link_type"] == "iot_host_match"]
    ca = links[links["link_type"] == "connect_accept"]
    ts = host["ts"].astype("datetime64[s]").astype("int64").to_numpy()
    hrow = host["row"].to_numpy()
    h_iot = ih["node_a"].str.split(":").str[1].astype(int).to_numpy()
    h_hop = ih["node_b"].str.split(":").str[1].astype(int).to_numpy()
    hop_host = host["host_ip"].to_numpy()[h_hop]
    hop_pid = host["process_id"].to_numpy()[h_hop]
    hop_ts = ts[h_hop]
    by_host = {}
    for i, h in enumerate(hop_host):
        by_host.setdefault(h, []).append(i)
    by_host = {h: np.array(sorted(ix, key=lambda i: hop_ts[i])) for h, ix in by_host.items()}

    conn = ca["node_a"].str.split(":").str[1].astype(int).to_numpy()
    acc = ca["node_b"].str.split(":").str[1].astype(int).to_numpy()
    out = []
    pid_arr, host_arr = host["process_id"].to_numpy(), host["host_ip"].to_numpy()
    for c, a in zip(conn, acc):
        cand = by_host.get(host_arr[c])
        if cand is None:
            continue
        times = hop_ts[cand]
        j = np.searchsorted(times, ts[c], side="right") - 1
        lin = lineage(int(pid_arr[c]))
        while j >= 0 and ts[c] - times[j] <= 1800:
            if int(hop_pid[cand[j]]) in lin:
                k = cand[j]
                out.append((n_host + int(h_iot[k]), int(h_hop[k]), int(c), int(a)))
                break
            j -= 1
    return pd.DataFrame(out, columns=["iot_flow", "hop_accept", "connect", "accept_b"])


def ssh_fragmentation(host, scenario):
    """For every SSH accept (port 22), the next process started by one of its descendants within 15 min:
    is it in the same scenario?"""
    ppid_of = dict(zip(host["process_id"].tolist(), host["parent_process_id"].tolist()))
    acc = host[(host["event_type"] == "network_accept") & (host["dst_port"] == "22")]
    ps = host[host["event_type"] == "process_start"].sort_values(["ts", "row"])
    ts = host["ts"].astype("datetime64[s]").astype("int64").to_numpy()
    by_host = {h: g for h, g in ps.groupby("host_ip")}
    same = total = 0
    for r in acc.itertuples():
        g = by_host.get(r.host_ip)
        if g is None:
            continue
        lo = np.searchsorted(ts[g["row"].to_numpy()], ts[r.row], side="left")
        for pid, prow, pts in zip(g["process_id"].to_numpy()[lo:lo + 60], g["row"].to_numpy()[lo:lo + 60],
                                  ts[g["row"].to_numpy()][lo:lo + 60]):
            if pts - ts[r.row] > 900:
                break
            cur, anc = pid, set()
            for _ in range(12):
                par = ppid_of.get(cur)
                if par is None or par in anc:
                    break
                anc.add(par)
                cur = par
            if r.process_id in anc:
                total += 1
                same += int(scenario[prow] == scenario[r.row])
                break
    return same, total


def auc(score, positive):
    """ROC-AUC of a single numeric feature (rank formula); 0.5 = no information."""
    s = pd.Series(score).rank(method="average")
    pos = np.asarray(positive, dtype=bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- naming (uses the label)
def scenario_table(nodes, scenario):
    df = nodes.assign(scenario=scenario)
    g = df.groupby("scenario")
    t = pd.DataFrame({"n_events": g.size(), "n_host": g["kind"].apply(lambda s: (s == "host").sum()),
                      "n_iot": g["kind"].apply(lambda s: (s == "iot").sum()),
                      "start_s": g["ts_s"].min(), "end_s": g["ts_s"].max(),
                      "n_hosts": g["host_ip"].apply(lambda s: s[s != ""].nunique()),
                      "n_label1": g["label"].sum()})
    t["duration_s"] = t["end_s"] - t["start_s"]
    t["label1_share"] = t["n_label1"] / t["n_events"]
    t["type"] = np.where(t["n_label1"] > 0, "attack-containing", "normal")
    t["has_cross_domain_link"] = (t["n_host"] > 0) & (t["n_iot"] > 0)
    return t.reset_index()


# --------------------------------------------------------------------------- extra link found by the chain audit
def handoff_links(host, services, gap=GATE_GAP):
    """Service hand-off: a long-lived service (web server, listener) accepts a request and the SAME process then
    connects to the next host. Linked to the nearest preceding accept by that process (<= 15 min).
    This treats a normal proxy chain exactly like the attack chains, where the shell started by the accepting
    process is linked the same way (entry_to_child)."""
    h = host.sort_values(["ts", "row"], kind="stable")
    conn = h[(h["event_type"] == "network_connect") & h["process_id"].isin(services) & (h["process_id"] != 1)]
    acc = h[h["event_type"] == "network_accept"]
    l = conn[["row", "ts", "host_ip", "process_id"]].rename(columns={"row": "conn_row", "ts": "conn_ts", "process_id": "_k"})
    r = acc[["row", "ts", "host_ip", "process_id"]].rename(columns={"row": "acc_row", "ts": "acc_ts", "process_id": "_k"})
    m = pd.merge_asof(l.sort_values("conn_ts", kind="stable"), r.sort_values("acc_ts", kind="stable"),
                      left_on="conn_ts", right_on="acc_ts", by=["host_ip", "_k"], direction="backward",
                      tolerance=pd.Timedelta(seconds=gap)).dropna(subset=["acc_row"])
    return pd.DataFrame({"node_a": "host:" + m["acc_row"].astype(int).astype(str).to_numpy(),
                         "node_b": "host:" + m["conn_row"].astype(str).to_numpy(),
                         "link_type": "service_handoff", "points": 2,
                         "gap_sec": (m["conn_ts"] - m["acc_ts"]).dt.total_seconds().to_numpy()})


def burst_cohesion(host, scenario, event_type, attack_type, min_size=6, gap=GATE_GAP):
    """Validation (uses `type`): rows of one attack type hitting one host from one source in a burst should
    land in one scenario. Returns (bursts, share of bursts with >=90% of their rows in the largest scenario)."""
    d = host[(host["event_type"] == event_type) & (host["type"] == attack_type)].sort_values(["host_ip", "src_ip", "ts"])
    bursts = whole = 0
    for _, g in d.groupby(["host_ip", "src_ip"]):
        t = g["ts"].astype("datetime64[s]").astype("int64").to_numpy()
        cut = np.r_[0, np.where(np.diff(t) > gap)[0] + 1, len(t)]
        rows = g["row"].to_numpy()
        for a, b in zip(cut[:-1], cut[1:]):
            if b - a >= min_size:
                sc = scenario[rows[a:b]]
                bursts += 1
                whole += int(np.bincount(sc).max() / (b - a) >= 0.9)
    return bursts, whole


def remote_source_links(host, gap=GATE_GAP):
    """Same remote source: consecutive accepts on the same host from the same remote address within 15 min
    (port scans, password guessing and floods are many connections from one source). Services handle these,
    so no process link exists between them; the shared communication endpoint is the evidence."""
    acc = host[(host["event_type"] == "network_accept") & (host["src_ip"] != "-")].sort_values(["ts", "row"], kind="stable")
    g = acc.groupby(["host_ip", "src_ip"], sort=False)
    prev = g["row"].shift()
    d = (acc["ts"] - g["ts"].shift()).dt.total_seconds()
    ok = prev.notna() & (d <= gap)
    return pd.DataFrame({"node_a": "host:" + prev[ok].astype(int).astype(str).to_numpy(),
                         "node_b": "host:" + acc["row"][ok].astype(str).to_numpy(),
                         "link_type": "same_remote_source", "points": 2, "gap_sec": d[ok].to_numpy()})


# =========================================================================== application integration
# Everything above this line is the Stage 3b notebook module, extracted verbatim (cell 6 of
# Stage3b_Scenario_Grouping_V6v2.ipynb). The code below is new and only wires the original routines together:
#   * label-free twins of `node_table` / `scenario_table` (the originals read the ground-truth `label` column);
#   * the notebook's orchestration (Steps 2, 3, 5, 6, 7 and 10) as functions;
#   * evaluation-only helpers that re-attach ground truth AFTER grouping, for reference comparison.
from dataclasses import dataclass

from . import relationship_mapping as rm

T_FINAL = 3                 # notebook Step 5/6: link set C, T = 3 selected from structural audits
MIN_SCENARIO_EVENTS = 2     # notebook Step 7: a graph needs >= 2 events (single events are set aside)
SUMMARY_STRUCTURAL_COLUMNS = ["scenario_id", "n_host", "n_iot", "n_events", "start", "end", "duration_s", "hosts",
                              "has_cross_domain_link"]
SUMMARY_LABEL_COLUMNS = ["n_label1", "label1_share", "type"]       # ground-truth derived: evaluation only
ASSIGNMENT_COLUMNS = ["node", "scenario_id"]


def node_table_structural(host, iot, managed_ips):
    """`node_table` without the ground-truth `label` column (otherwise identical)."""
    n_host = len(host)
    iot_host_ip = np.where(iot["src_ip"].isin(managed_ips), iot["src_ip"],
                           np.where(iot["dst_ip"].isin(managed_ips), iot["dst_ip"], ""))
    return pd.DataFrame({
        "node": np.concatenate(["host:" + host["row"].astype(str).to_numpy(), "iot:" + iot["row"].astype(str).to_numpy()]),
        "kind": ["host"] * n_host + ["iot"] * len(iot),
        "ts_s": np.concatenate([host["ts"].astype("datetime64[s]").astype("int64").to_numpy(),
                                iot["ts"].astype("datetime64[s]").astype("int64").to_numpy()]).astype(np.int64),
        "host_ip": np.concatenate([host["host_ip"].to_numpy(object), iot_host_ip.astype(object)]),
    })


def scenario_table_structural(nodes, scenario):
    """`scenario_table` without n_label1 / label1_share / type (otherwise identical)."""
    df = nodes.assign(scenario=scenario)
    g = df.groupby("scenario")
    t = pd.DataFrame({"n_events": g.size(), "n_host": g["kind"].apply(lambda s: (s == "host").sum()),
                      "n_iot": g["kind"].apply(lambda s: (s == "iot").sum()),
                      "start_s": g["ts_s"].min(), "end_s": g["ts_s"].max(),
                      "n_hosts": g["host_ip"].apply(lambda s: s[s != ""].nunique())})
    t["duration_s"] = t["end_s"] - t["start_s"]
    t["has_cross_domain_link"] = (t["n_host"] > 0) & (t["n_iot"] > 0)
    return t.reset_index()


def combined_links(data, stage3_links=None):
    """Stage 3 links + service_handoff + same_remote_source (link set C), exactly as in the notebook (Step 5)."""
    host, _ = data.to_stage3_frames()
    if stage3_links is None:
        stage3_links = rm.build_links_from_stage2(data)
    services = rm.service_pids(host)
    return pd.concat([stage3_links, handoff_links(host, services), remote_source_links(host)], ignore_index=True)


def _summary(tbl, nodes, scenario):
    """Notebook Step 10 summary construction; label columns are included only when `tbl` has them."""
    inc = tbl[tbl["n_events"] >= MIN_SCENARIO_EVENTS]
    has_host = (nodes["host_ip"] != "").to_numpy()
    hosts_of = (nodes[has_host].assign(scenario=scenario[has_host])
                .groupby("scenario")["host_ip"].apply(lambda s: ";".join(sorted(set(s)))))
    summary = inc.copy()
    summary["start"] = pd.to_datetime(summary["start_s"], unit="s").dt.strftime("%d/%m/%Y %H:%M:%S")
    summary["end"] = pd.to_datetime(summary["end_s"], unit="s").dt.strftime("%d/%m/%Y %H:%M:%S")
    summary["hosts"] = summary["scenario"].map(hosts_of).fillna("")
    summary = summary.rename(columns={"scenario": "scenario_id"})
    cols = SUMMARY_STRUCTURAL_COLUMNS + (SUMMARY_LABEL_COLUMNS if "type" in summary.columns else [])
    summary = summary[cols]
    if "label1_share" in summary.columns:
        summary = summary.round({"label1_share": 3})
    return summary


@dataclass
class Stage3bResult:
    """Label-free grouping result. Row identifiers are the original 0-based file rows (`host:<row>` / `iot:<row>`)."""
    links: pd.DataFrame          # combined link set C (node_a, node_b, link_type, points, gap_sec)
    nodes: pd.DataFrame          # node, kind, ts_s, host_ip - host rows first, then iot rows
    pairs: pd.DataFrame          # scored, time-gated linked pairs
    scenario: np.ndarray         # scenario id per node (aligned with `nodes`); single events get their own id
    scenarios: pd.DataFrame      # structural table of ALL scenarios, including single-event ones
    assignments: pd.DataFrame    # node, scenario_id for every event
    summary: pd.DataFrame        # structural summary of scenarios with >= 2 events
    threshold: int

    def stage4_inputs(self):
        """The label-free tables Stage 4 reads: event -> scenario assignment, scenario summary and the links."""
        return {"assign": self.assignments, "summ": self.summary, "links": self.links}


def run_stage3b(data, T=T_FINAL, stage3_links=None) -> Stage3bResult:
    """Stage 2 data -> Stage 3 links -> link set C -> pair scores -> scenarios (T = 3, 15-minute continuity)."""
    host, iot = data.to_stage3_frames()
    managed_ips = set(host["host_ip"])
    links = combined_links(data, stage3_links)
    nodes = node_table_structural(host, iot, managed_ips)
    pairs = pair_scores(links, nodes, len(host))
    scenario = group_scenarios(pairs, len(nodes), nodes["ts_s"].to_numpy(), T)
    tbl = scenario_table_structural(nodes, scenario)
    assignments = pd.DataFrame({"node": nodes["node"], "scenario_id": scenario})
    return Stage3bResult(links, nodes, pairs, scenario, tbl, assignments, _summary(tbl, nodes, scenario), T)


# --------------------------------------------------------------------------- evaluation only (uses ground truth)
def node_labels_for_evaluation(data):
    """Ground-truth event labels in node order (host rows, then iot rows). NEVER pass these into grouping."""
    return np.concatenate([data.host_truth["label"].astype(int).to_numpy(),
                           data.iot_truth["label"].astype(int).to_numpy()])


def evaluation_tables(result, data):
    """Re-create the notebook's labelled `scenario_assignments` and `scenario_summary` for comparison with the
    saved references. Uses the ORIGINAL `scenario_table`; grouping is already finished and is not affected."""
    nodes_eval = result.nodes.assign(label=node_labels_for_evaluation(data))
    tbl = scenario_table(nodes_eval, result.scenario)
    assign = result.assignments.merge(tbl[["scenario", "type"]], left_on="scenario_id",
                                      right_on="scenario").drop(columns="scenario")
    return assign, _summary(tbl, nodes_eval, result.scenario)
