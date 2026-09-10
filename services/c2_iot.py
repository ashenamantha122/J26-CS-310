"""
Component 2 -- Agentless IoT network telemetry anomaly detection.
IT23442320 | Jayathilaka N.M.O.R

Novelty implemented here: five adversarially-controllable identifiers are
excluded on threat-model grounds BEFORE any statistical feature selection or
training. The exclusion is a property of the threat model, not of the dataset,
so it transfers to networks the model has never seen.

Set A = identifier-inclusive (the baseline everyone else trains)
Set B = behavioural-only     (the proposed feature set)

The evaluation that matters is not accuracy on a clean holdout -- Set A usually
wins there. It is what survives when the attacker perturbs the identifiers they
control. T2 below measures exactly that.

Model is sklearn GradientBoosting so the prototype runs with no extra installs;
LightGBM drops in at the marked line with identical semantics.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from laax.bus import Bus
from laax.contracts import FlowVerdict, Topic

# --------------------------------------------------------------------------
# The threat model, stated as data.
#
# An attacker who controls a compromised IoT device can set every one of these
# at will. A model that leans on them is measuring the attacker's choices.
# --------------------------------------------------------------------------
CONTROLLABLE_IDENTIFIERS = [
    "src_ip_int",
    "dst_ip_int",
    "src_port",
    "dst_port",
    "dns_query_hash",
]

# Behavioural features: statistical properties of the flow that arise from what
# the traffic *does*, which an attacker cannot set without changing behaviour.
BEHAVIOURAL_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "bytes_ratio",
    "mean_iat",
    "std_iat",
    "pkt_size_entropy",
    "proto_tcp",
]

FEATURE_SETS = {
    "A": BEHAVIOURAL_FEATURES + CONTROLLABLE_IDENTIFIERS,
    "B": BEHAVIOURAL_FEATURES,
}


@dataclass
class Flow:
    device_id: str
    saddr: str
    daddr: str
    dport: int
    proto: str
    features: Dict[str, float]
    label: int          # 0 benign, 1 malicious
    label_name: str


# --------------------------------------------------------------------------
def _behaviour(rng: random.Random, malicious: bool) -> Dict[str, float]:
    """Behavioural half of a flow record. Identifiers are added separately."""
    if malicious:
        duration = rng.expovariate(1 / 0.35)
        src_pkts = rng.randint(1, 14)
        dst_pkts = rng.randint(0, 8)
        src_bytes = src_pkts * rng.randint(40, 260)
        dst_bytes = dst_pkts * rng.randint(40, 200)
        mean_iat = rng.uniform(0.001, 0.30)
        std_iat = mean_iat * rng.uniform(0.02, 0.9)       # machine-regular
        entropy = rng.uniform(0.5, 4.2)
    else:
        duration = rng.expovariate(1 / 4.0)
        src_pkts = rng.randint(2, 60)
        dst_pkts = rng.randint(2, 60)
        src_bytes = src_pkts * rng.randint(60, 900)
        dst_bytes = dst_pkts * rng.randint(60, 1400)
        mean_iat = rng.uniform(0.01, 1.2)
        std_iat = mean_iat * rng.uniform(0.25, 1.6)       # human-irregular
        entropy = rng.uniform(2.4, 6.5)
    return {
        "duration": duration,
        "src_bytes": float(src_bytes),
        "dst_bytes": float(dst_bytes),
        "src_pkts": float(src_pkts),
        "dst_pkts": float(dst_pkts),
        "bytes_ratio": src_bytes / max(dst_bytes, 1.0),
        "mean_iat": mean_iat,
        "std_iat": std_iat,
        "pkt_size_entropy": entropy,
        "proto_tcp": 1.0,
    }


def synth_flows(n: int, seed: int = 0, attack_ratio: float = 0.3,
                network: str = "ton_iot", mimicry: float = 0.35) -> List[Flow]:
    """
    TON_IoT-shaped synthetic flows.

    Two properties are deliberate, because they are what makes the evaluation
    say something:

    1. Identifiers are perfectly predictive WITHIN a network -- attacks come
       from one address block and hit a few ports -- but the block differs per
       network. This is the dataset bias a Set A model happily learns.
    2. A `mimicry` fraction of attacks behave like benign traffic. Those flows
       are separable ONLY by identifier. They are the reason Set A scores higher
       on a clean holdout, and excluding identifiers has a real cost that the
       proposal commits to quantifying rather than hiding.
    """
    rng = random.Random(seed)
    atk_net = {"ton_iot": 66, "iot_23": 203}[network]
    ben_net = {"ton_iot": 10, "iot_23": 192}[network]
    out: List[Flow] = []
    for _ in range(n):
        mal = rng.random() < attack_ratio
        # A mimicking attacker shapes traffic to look ordinary; only the
        # identifiers still betray it.
        feats = _behaviour(rng, malicious=mal and rng.random() >= mimicry)
        if mal:
            sip = f"{atk_net}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
            dport = rng.choice([23, 2323, 445, 22])
            name = rng.choice(["ddos", "scanning", "backdoor", "injection"])
        else:
            sip = f"{ben_net}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
            dport = rng.choice([80, 443, 1883, 8883, 502])
            name = "normal"
        dip = f"{ben_net}.0.0.{rng.randint(2, 60)}"
        sport = rng.randint(1024, 65535)
        feats.update({
            "src_ip_int": float(int(sip.split(".")[0]) * 256 + int(sip.split(".")[1])),
            "dst_ip_int": float(int(dip.split(".")[0]) * 256 + int(dip.split(".")[1])),
            "src_port": float(sport),
            "dst_port": float(dport),
            "dns_query_hash": float(hash((name, atk_net if mal else ben_net)) % 10007),
        })
        out.append(Flow(f"iot-{rng.randint(1, 12):02d}", sip, dip, dport, "tcp",
                        feats, int(mal), name))
    return out


def _matrix(flows: List[Flow], feature_set: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = FEATURE_SETS[feature_set]
    X = np.array([[f.features[c] for c in cols] for f in flows], dtype=float)
    y = np.array([f.label for f in flows], dtype=int)
    return X, y


def perturb_identifiers(flows: List[Flow], seed: int = 7) -> List[Flow]:
    """
    T2: the attacker rerolls every identifier they control, and nothing else.
    Behaviour is untouched, so a behaviour-based detector must be unaffected.
    """
    rng = random.Random(seed)
    out = []
    for f in flows:
        g = Flow(**{**f.__dict__, "features": dict(f.features)})
        for k in CONTROLLABLE_IDENTIFIERS:
            if k == "src_port":
                g.features[k] = float(rng.randint(1024, 65535))
            elif k == "dst_port":
                g.features[k] = float(rng.choice([80, 443, 8080, 53, 1883]))
            elif k == "dns_query_hash":
                g.features[k] = float(rng.randint(0, 10007))
            else:
                g.features[k] = float(rng.randint(0, 65535))
        out.append(g)
    return out


# --------------------------------------------------------------------------
class C2Classifier:
    def __init__(self, feature_set: str = "B", model: str = "gbdt") -> None:
        self.feature_set = feature_set
        self.model_name = model
        # Swap for lightgbm.LGBMClassifier(n_estimators=200, num_leaves=31) here.
        self.clf = (GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
                    if model == "gbdt" else
                    DecisionTreeClassifier(max_depth=8, random_state=0))

    def fit(self, flows: List[Flow]) -> "C2Classifier":
        X, y = _matrix(flows, self.feature_set)
        self.clf.fit(X, y)
        return self

    def macro_f1(self, flows: List[Flow]) -> float:
        X, y = _matrix(flows, self.feature_set)
        return float(f1_score(y, self.clf.predict(X), average="macro"))

    def score_one(self, flow: Flow) -> float:
        cols = FEATURE_SETS[self.feature_set]
        x = np.array([[flow.features[c] for c in cols]], dtype=float)
        return float(self.clf.predict_proba(x)[0, 1])


def run_evaluation(verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """T1 clean holdout, T2 identifier perturbation, T3 cross-dataset transfer."""
    train = synth_flows(2400, seed=1, network="ton_iot")
    test = synth_flows(800, seed=2, network="ton_iot")
    test_perturbed = perturb_identifiers(test)
    transfer = synth_flows(800, seed=3, network="iot_23")   # different address plan

    results: Dict[str, Dict[str, float]] = {}
    for fs in ("A", "B"):
        clf = C2Classifier(feature_set=fs).fit(train)
        t1 = clf.macro_f1(test)
        t2 = clf.macro_f1(test_perturbed)
        t3 = clf.macro_f1(transfer)
        results[fs] = {
            "T1_clean": round(t1, 4),
            "T2_perturbed": round(t2, 4),
            "T2_drop": round(t1 - t2, 4),
            "T3_transfer": round(t3, 4),
        }
    if verbose:
        print("  Set A = identifier-inclusive baseline | Set B = behavioural-only (proposed)")
        print(f"  {'':<14}{'T1 clean':>10}{'T2 perturb':>12}{'drop':>9}{'T3 transfer':>13}")
        for fs, r in results.items():
            print(f"  Set {fs:<10}{r['T1_clean']:>10.4f}{r['T2_perturbed']:>12.4f}"
                  f"{r['T2_drop']:>9.4f}{r['T3_transfer']:>13.4f}")
    return results


# --------------------------------------------------------------------------
class C2Service:
    """Scores a replayed flow stream and publishes verdicts to C3."""

    def __init__(self, bus: Bus, clf: C2Classifier,
                 suspicious: float = 0.5, malicious: float = 0.8) -> None:
        self.bus, self.clf = bus, clf
        self.suspicious, self.malicious = suspicious, malicious
        self.published = 0

    async def score(self, flow: Flow) -> FlowVerdict:
        p = self.clf.score_one(flow)
        verdict = ("malicious" if p >= self.malicious
                   else "suspicious" if p >= self.suspicious else "benign")
        msg = FlowVerdict(
            device_id=flow.device_id, saddr=flow.saddr, daddr=flow.daddr,
            dport=flow.dport, proto=flow.proto, score=round(p, 4),
            verdict=verdict, feature_set=self.clf.feature_set,
            model=self.clf.model_name, label=flow.label_name,
        )
        await self.bus.publish(Topic.FLOW_VERDICT, msg)
        self.published += 1
        if verdict != "benign":
            print(f"  [C2] {verdict.upper():10} {flow.device_id} -> {flow.daddr}:{flow.dport} "
                  f"p={p:.3f} (set {self.clf.feature_set}, truth={flow.label_name})")
        return msg
