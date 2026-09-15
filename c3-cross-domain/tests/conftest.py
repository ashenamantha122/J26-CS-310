"""Session-scoped real-data fixtures shared by the graph-builder tests (the Stage 2 -> 3b run takes ~1 min)."""
import numpy as np
import pytest

from xdr_c3 import f1_f2_handoff as h
from xdr_c3 import graph_builder as gb
from xdr_c3 import scenario_grouping as sg
from xdr_c3 import stage2_validate as s2
from xdr_c3.model_contract import load_contract


@pytest.fixture(scope="session")
def s2_data():
    return s2.load_and_validate()


@pytest.fixture(scope="session")
def s3b(s2_data):
    return sg.run_stage3b(s2_data)


@pytest.fixture(scope="session")
def contract():
    return load_contract()


@pytest.fixture(scope="session")
def graphs(s2_data, s3b, contract):
    return gb.build_graphs_from_stage3b(s2_data, s3b, contract)


@pytest.fixture(scope="session")
def detector():
    from xdr_c3.inference import Stage5Detector
    return Stage5Detector.load(use_cache=False)         # rebuilds the 8,946 normal reference embeddings (~3 s)


@pytest.fixture(scope="session")
def result(detector, graphs):
    return detector.predict_collection(graphs)


@pytest.fixture(scope="session")
def handoff(s2_data, s3b, result, detector):
    return h.build_f1_f2_handoff(s2_data, s3b, result, detector.contract)


@pytest.fixture(scope="session")
def mixed_ids(result):
    """A reproducible real sample: 12 Suspicious and 12 Normal scenario ids from the actual model decisions."""
    s = result.scores
    return np.sort(np.r_[s.loc[s.is_suspicious, "scenario_id"].to_numpy()[::120][:12],
                         s.loc[~s.is_suspicious, "scenario_id"].to_numpy()[::900][:12]])


@pytest.fixture(scope="session")
def small_result(detector, graphs, mixed_ids):
    return detector.predict_collection(graphs, mixed_ids)


