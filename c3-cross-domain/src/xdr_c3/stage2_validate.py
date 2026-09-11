"""Stage 2 - input validation for the Host / IoT telemetry (application version of the Stage 2 notebook).

The rules, column lists, column roles and the strict timestamp format are taken from
``Stage2_Data_Preprocessing_V6v2_2019.ipynb``. Nothing is changed in the data: both files are read as text
(``dtype=str, keep_default_na=False``), ``-`` and blank are kept as "not applicable" (never imputed), and the
original column order is preserved.

Outputs are in-memory and compatible with the Stage 3 ``load_data`` result (``ts`` and ``row`` added,
``process_id`` / ``parent_process_id`` as int64), with one difference by design: the ground-truth columns
(host ``label`` / ``type``, IoT ``label`` / ``attack_type``; role "never a feature") are held in separate
frames, so the inference frames cannot leak them. ``to_stage3_frames(include_ground_truth=True)`` rebuilds the
exact Stage 3 layout for evaluation-only code.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HOST_CSV, IOT_CSV, TIMESTAMP_FORMAT

FMT = TIMESTAMP_FORMAT

# --------------------------------------------------------------------------- schema (Stage 2 cell 6)
HOST_COLS = ["timestamp", "host_id", "host_ip", "process_id", "parent_process_id", "process_name", "event_type",
             "trace_mode", "trace_source", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "syscall_name",
             "syscall_result", "process_state", "memory_alloc_bytes", "file_operation", "threat_confidence_score",
             "threat_state", "cpu_usage", "disk_read_bytes", "disk_write_bytes", "label", "type"]
IOT_COLS = ["timestamp", "iot_device_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "service",
            "flow_duration", "src_bytes", "dst_bytes", "src_packets", "dst_packets", "iat_mean", "iat_std",
            "packet_size_mean", "packet_size_std", "connection_rate", "confidence_score", "label",
            "predicted_class", "attack_type"]

# Numeric columns checked for parsing / negatives (Stage 2 cell 9)
HOST_NUM = ["process_id", "parent_process_id", "memory_alloc_bytes", "threat_confidence_score", "cpu_usage",
            "disk_read_bytes", "disk_write_bytes", "label"]
IOT_NUM = ["src_port", "dst_port", "flow_duration", "src_bytes", "dst_bytes", "src_packets", "dst_packets",
           "iat_mean", "iat_std", "packet_size_mean", "packet_size_std", "connection_rate", "confidence_score", "label"]

# Column roles (Stage 2 cell 16)
HOST_ROLES = {
    "timestamp": "structure", "host_id": "identity", "host_ip": "structure", "process_id": "structure",
    "parent_process_id": "structure", "process_name": "categorical feature", "event_type": "categorical feature",
    "trace_mode": "component output", "trace_source": "categorical feature", "src_ip": "structure",
    "dst_ip": "structure", "src_port": "structure", "dst_port": "structure", "protocol": "categorical feature",
    "syscall_name": "categorical feature", "syscall_result": "categorical feature",
    "process_state": "categorical feature", "memory_alloc_bytes": "numeric feature",
    "file_operation": "categorical feature", "threat_confidence_score": "component output",
    "threat_state": "component output", "cpu_usage": "numeric feature", "disk_read_bytes": "numeric feature",
    "disk_write_bytes": "numeric feature", "label": "never a feature", "type": "never a feature"}
IOT_ROLES = {
    "timestamp": "structure", "iot_device_id": "identity", "src_ip": "structure", "dst_ip": "structure",
    "src_port": "structure", "dst_port": "structure", "protocol": "categorical feature",
    "service": "categorical feature", "flow_duration": "numeric feature", "src_bytes": "numeric feature",
    "dst_bytes": "numeric feature", "src_packets": "numeric feature", "dst_packets": "numeric feature",
    "iat_mean": "numeric feature", "iat_std": "numeric feature", "packet_size_mean": "numeric feature",
    "packet_size_std": "numeric feature", "connection_rate": "numeric feature",
    "confidence_score": "component output", "label": "never a feature", "predicted_class": "component output",
    "attack_type": "never a feature"}
assert list(HOST_ROLES) == HOST_COLS and list(IOT_ROLES) == IOT_COLS

HOST_TRUTH_COLS = [c for c, r in HOST_ROLES.items() if r == "never a feature"]   # label, type
IOT_TRUTH_COLS = [c for c, r in IOT_ROLES.items() if r == "never a feature"]     # label, attack_type

NET_EVENTS = ["network_connect", "network_accept"]                               # Stage 2 / Stage 3
TUPLE_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
BANNED_CATEGORIES = ["lateral_movement", "unauthorized_access", "scan"]          # not row-level categories
PACKET_SIZE_TOL = 0.01                                                           # notebook: max diff 0.005 (rounding)


class Stage2ValidationError(ValueError):
    """Raised when the input fails a mandatory Stage 2 check."""

    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__("Stage 2 validation failed:\n" + "\n".join(f"  - {c.name}: {c.detail}" for c in report.errors))


# --------------------------------------------------------------------------- report
@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str          # "error" fails validation, "warning" is reported only
    detail: str = ""
    value: object = None


@dataclass
class ValidationReport:
    checks: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, name, passed, detail="", severity="error", value=None):
        self.checks.append(CheckResult(name, bool(passed), severity, detail, value))

    @property
    def errors(self):
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self):
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def ok(self):
        return not self.errors

    def summary(self):
        lines = [f"Stage 2 validation: {'PASS' if self.ok else 'FAIL'} "
                 f"({len(self.checks)} checks, {len(self.errors)} errors, {len(self.warnings)} warnings)"]
        for c in self.checks:
            mark = "ok  " if c.passed else ("WARN" if c.severity == "warning" else "FAIL")
            lines.append(f"  [{mark}] {c.name}" + (f" - {c.detail}" if c.detail else ""))
        return "\n".join(lines)


@dataclass
class Stage2Data:
    """Validated telemetry. ``host`` / ``iot`` hold the inference-side columns plus ``ts`` and ``row``;
    ``host_truth`` / ``iot_truth`` hold the ground-truth columns (raw text) indexed by ``row``."""
    host: pd.DataFrame
    iot: pd.DataFrame
    host_truth: pd.DataFrame
    iot_truth: pd.DataFrame
    report: ValidationReport

    def to_stage3_frames(self, include_ground_truth: bool = False):
        """Frames in the Stage 3 ``load_data`` layout. With ``include_ground_truth=True`` the result is identical
        (columns, order, dtypes, values) to ``relationship_mapping.load_data``; use that for evaluation only."""
        if not include_ground_truth:
            return self.host.copy(), self.iot.copy()
        return (_with_truth(self.host, self.host_truth, HOST_COLS),
                _with_truth(self.iot, self.iot_truth, IOT_COLS))


def _with_truth(feat, truth, cols):
    out = pd.concat([feat.reset_index(drop=True), truth.reset_index(drop=True)], axis=1)
    return out[cols + ["ts", "row"]]


# --------------------------------------------------------------------------- helpers (Stage 2 cell 9)
def _bad_ips(s):
    bad = 0
    for v in s.unique():
        if v in ("-", ""):
            continue
        try:
            ipaddress.ip_address(v)
        except ValueError:
            bad += 1
    return bad


def _num(df, c):
    return pd.to_numeric(df[c].replace("", np.nan), errors="coerce")


def read_raw(host_path=HOST_CSV, iot_path=IOT_CSV):
    """Read both files exactly as text; placeholders ('-' and blank) are preserved."""
    host = pd.read_csv(Path(host_path), dtype=str, keep_default_na=False)
    iot = pd.read_csv(Path(iot_path), dtype=str, keep_default_na=False)
    return host, iot


def expected_columns(kind, allow_missing_ground_truth=False):
    cols, truth = (HOST_COLS, HOST_TRUTH_COLS) if kind == "host" else (IOT_COLS, IOT_TRUTH_COLS)
    return [c for c in cols if c not in truth] if allow_missing_ground_truth else list(cols)


# --------------------------------------------------------------------------- validation
def validate_frames(host, iot, allow_missing_ground_truth=False) -> ValidationReport:
    """Run the Stage 2 checks on raw (all-text) frames. Inputs are not modified."""
    rep = ValidationReport()

    # schema: exact column names and order
    schema_ok = True
    for name, df, kind in (("host", host, "host"), ("iot", iot, "iot")):
        exp = expected_columns(kind, allow_missing_ground_truth)
        ok = list(df.columns) == exp
        schema_ok &= ok
        rep.add(f"{name} columns exactly as expected", ok,
                "" if ok else f"missing={[c for c in exp if c not in df.columns]} "
                              f"extra={[c for c in df.columns if c not in exp]} (order must also match)")
    if not schema_ok:
        return rep
    rep.stats.update(host_rows=len(host), host_columns=host.shape[1], iot_rows=len(iot), iot_columns=iot.shape[1])

    # all values must be text (raw read)
    for name, df in (("host", host), ("iot", iot)):
        bad = [c for c in df.columns if not all(isinstance(v, str) for v in df[c].head(1000))]
        rep.add(f"{name} read as text", not bad, f"non-text columns: {bad}" if bad else "")

    # strict timestamps
    ts = {}
    for name, df in (("host", host), ("iot", iot)):
        try:
            ts[name] = pd.to_datetime(df["timestamp"], format=FMT)
            # pandas turns an empty string into NaT without raising; the notebook's "strict" intent forbids it
            n_nat = int(ts[name].isna().sum())
            rep.add(f"{name} timestamps parse as {FMT}", n_nat == 0,
                    f"{ts[name].min()} -> {ts[name].max()}" if n_nat == 0 else f"{n_nat} blank/NaT timestamps")
        except (ValueError, TypeError) as e:
            rep.add(f"{name} timestamps parse as {FMT}", False, str(e).splitlines()[0])
    if len(ts) < 2:
        return rep
    for name in ("host", "iot"):
        srt = bool(ts[name].is_monotonic_increasing)
        rep.add(f"{name} timestamps in time order", srt, "" if srt else "not sorted (Stage 3 sorts internally)",
                severity="warning")

    # process ids must be integers (Stage 3 load_data casts them to int64)
    for c in ("process_id", "parent_process_id"):
        ok = bool(host[c].str.fullmatch(r"-?\d+").all())
        rep.add(f"host {c} is integer text", ok)

    # quality checks (Stage 2 cell 9)
    dup = (int(host.duplicated().sum()), int(iot.duplicated().sum()))
    rep.add("no duplicate rows (host / iot)", dup == (0, 0), f"{dup}", value=dup)
    ips = (_bad_ips(host["src_ip"]) + _bad_ips(host["dst_ip"]), _bad_ips(iot["src_ip"]) + _bad_ips(iot["dst_ip"]))
    rep.add("no invalid IP strings (host / iot)", ips == (0, 0), f"{ips}", value=ips)
    h_num = [c for c in HOST_NUM if c in host.columns]     # 'label' absent when ground truth is not supplied
    i_num = [c for c in IOT_NUM if c in iot.columns]
    nan = (sum(int(_num(host, c).isna().sum()) for c in h_num), sum(int(_num(iot, c).isna().sum()) for c in i_num))
    rep.add("numeric columns all parse (host / iot)", nan == (0, 0), f"unparseable cells {nan}", value=nan)
    neg = (sum(int((_num(host, c) < 0).sum()) for c in h_num), sum(int((_num(iot, c) < 0).sum()) for c in i_num))
    rep.add("no negative numbers (host / iot)", neg == (0, 0), f"{neg}", value=neg)
    hp = pd.concat([_num(host, "src_port"), _num(host, "dst_port")]).dropna()
    ip_ = pd.concat([_num(iot, "src_port"), _num(iot, "dst_port")])
    ports = (int(((hp < 0) | (hp > 65535)).sum()), int(((ip_ < 0) | (ip_ > 65535)).sum()))
    rep.add("ports inside 0-65535 (host / iot)", ports == (0, 0), f"{ports}", value=ports)
    hs, is_ = _num(host, "threat_confidence_score"), _num(iot, "confidence_score")
    scores = (int(((hs < 0) | (hs > 1)).sum()), int(((is_ < 0) | (is_ > 1)).sum()))
    rep.add("scores inside [0,1] (host / iot)", scores == (0, 0), f"{scores}", value=scores)

    # IoT arithmetic: mean packet size = total bytes / total packets (rounding only)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = (_num(iot, "packet_size_mean") - (_num(iot, "src_bytes") + _num(iot, "dst_bytes")) /
                (_num(iot, "src_packets") + _num(iot, "dst_packets"))).abs()
    md = float(diff[np.isfinite(diff)].max()) if np.isfinite(diff).any() else 0.0
    rep.add("iot packet_size_mean = bytes/packets (rounding)", md <= PACKET_SIZE_TOL, f"max abs diff {md:.4f}",
            severity="warning", value=md)

    # placeholders (Stage 2 cell 12): the tuple is complete on network events and absent on all others
    net = host["event_type"].isin(NET_EVENTS)
    tup = host.loc[net, TUPLE_COLS]
    incomplete = int(((tup == "").any(axis=1) | (host.loc[net, ["src_ip", "dst_ip", "protocol"]] == "-").any(axis=1)).sum())
    rep.add("network events carry a complete tuple", incomplete == 0, f"{incomplete} incomplete of {int(net.sum())}",
            value=incomplete)
    carrying = int((host.loc[~net, ["src_ip", "dst_ip", "protocol"]] != "-").any(axis=1).sum())
    rep.add("non-network events carry no tuple", carrying == 0, f"{carrying} carrying", value=carrying)
    mem_ok = bool(((_num(host, "memory_alloc_bytes") > 0) == (host["event_type"] == "memory_event")).all())
    rep.add("memory_alloc_bytes > 0 only on memory events", mem_ok, severity="warning")
    iot_ph = int(((iot == "-") | (iot == "")).sum().sum())
    rep.add("iot has no placeholder cells", iot_ph == 0, f"{iot_ph} cells", severity="warning", value=iot_ph)

    # ground truth / categories (Stage 2 cell 9)
    if not allow_missing_ground_truth:
        bad_lab = (int((~host["label"].isin(["0", "1"])).sum()), int((~iot["label"].isin(["0", "1"])).sum()))
        rep.add("label values only 0/1 (host / iot)", bad_lab == (0, 0), f"{bad_lab}", value=bad_lab)
        consistent = bool(((host["label"] == "0") == (host["type"] == "normal")).all())
        rep.add("host type == 'normal' exactly when label == 0", consistent)
        banned = [c for c in BANNED_CATEGORIES if (host["type"] == c).any() or (iot["attack_type"] == c).any()]
        rep.add("no banned row-level categories", not banned, f"present: {banned}" if banned else "")
    rep.add("no scenario_id column", "scenario_id" not in host.columns and "scenario_id" not in iot.columns)
    return rep


def _build(host_raw, iot_raw, report, allow_missing_ground_truth):
    """Stage 3 ``load_data`` equivalent, with ground truth split off. Inputs are not modified."""
    host, iot = host_raw.copy(), iot_raw.copy()
    host["ts"] = pd.to_datetime(host["timestamp"], format=FMT)
    iot["ts"] = pd.to_datetime(iot["timestamp"], format=FMT)
    host["row"] = np.arange(len(host))                    # 0-based position in the original file
    iot["row"] = np.arange(len(iot))
    for c in ["process_id", "parent_process_id"]:
        host[c] = host[c].astype("int64")
    h_truth = [c for c in HOST_TRUTH_COLS if c in host.columns]
    i_truth = [c for c in IOT_TRUTH_COLS if c in iot.columns]
    host_truth = host[h_truth].copy()
    iot_truth = iot[i_truth].copy()
    host_truth.index = pd.Index(host["row"].to_numpy(), name="row")
    iot_truth.index = pd.Index(iot["row"].to_numpy(), name="row")
    return Stage2Data(host.drop(columns=h_truth), iot.drop(columns=i_truth), host_truth, iot_truth, report)


def validate_and_build(host_raw, iot_raw, allow_missing_ground_truth=False, raise_on_error=True) -> Stage2Data:
    report = validate_frames(host_raw, iot_raw, allow_missing_ground_truth)
    if not report.ok:
        if raise_on_error:
            raise Stage2ValidationError(report)
        return Stage2Data(None, None, None, None, report)
    return _build(host_raw, iot_raw, report, allow_missing_ground_truth)


def load_and_validate(host_path=HOST_CSV, iot_path=IOT_CSV, allow_missing_ground_truth=False,
                      raise_on_error=True) -> Stage2Data:
    """Read the two CSVs (read-only), run the Stage 2 checks and return validated in-memory data."""
    host_raw, iot_raw = read_raw(host_path, iot_path)
    return validate_and_build(host_raw, iot_raw, allow_missing_ground_truth, raise_on_error)


def assert_no_ground_truth(df):
    """Guard for model-input frames: raises if a ground-truth column is present."""
    leaked = [c for c in df.columns if c in set(HOST_TRUTH_COLS) | set(IOT_TRUTH_COLS)]
    if leaked:
        raise ValueError(f"ground-truth columns must not be inference inputs: {leaked}")
    return df
