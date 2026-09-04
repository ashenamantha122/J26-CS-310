# Component 3: Cross-Domain Correlation Engine (C3)

Lightweight Agent-Based & Agentless XDR Framework for Hybrid Enterprise-IoT Environments (Project **J26-CS-310**).

Component 3 performs cross-domain entity resolution, multi-partite provenance graph construction, and anomaly detection using Relational Graph Convolutional Networks (R-GCN) and Graph Autoencoders (GAE) to detect multi-stage lateral movement attacks spanning IoT network perimeters and enterprise host infrastructure.

## Research Architecture & Pipeline

The C3 pipeline consists of five progressive stages:

1. **Stage 1 / Ingestion**: Ingestion of host telemetry from Component 1 (eBPF-derived process, socket, and syscall events conforming to the 26-column schema) and Component 2 (IoT network flows, TON_IoT format).
2. **Stage 2: Data Preprocessing & Validation (`xdr_c3.stage2_validate`)**:
   - Strict timestamp parsing (`%d/%m/%Y %H:%M:%S`), schema validation, and sanitization.
   - Temporal boundary verification across host and network telemetry.
3. **Stage 3: Cross-Domain Relationship Mapping (`xdr_c3.relationship_mapping`)**:
   - Entity linkage between host sockets and network flows via 5-tuple matching (IP, port, protocol).
   - Temporal window synthesis linking IoT devices to host processes.
4. **Stage 3b: Scenario Grouping & Temporal Windowing (`xdr_c3.scenario_grouping`)**:
   - Sliding-window incident candidate clustering.
   - Attack scenario partition and threshold calibration.
5. **Stage 4: Multi-partite Graph Construction (`xdr_c3.graph_builder`)**:
   - Heterogeneous node typing (`process`, `socket`, `ip`, `iot_device`, `host`).
   - Typed relational edges (`spawn`, `socket_bind`, `network_flow`, `subnetwork`).
   - Graph feature matrix extraction, normalization, and split generation (Train, Validation, Test-Normal, Test-Attack).
6. **Stage 5: R-GCN & GAE Link Prediction & Anomaly Scoring (`xdr_c3.stage5_model`, `xdr_c3.stage5_utils`)**:
   - Heterogeneous message-passing architecture over multi-relational graphs.
   - Structural reconstruction error minimization for anomaly scoring.
   - Model checkpointing and locked selection (`epoch_040.pt`).
7. **Real-time Correlation & Lateral Movement Inference (`xdr_c3.inference`, `xdr_c3.lateral_movement`)**:
   - Lateral movement path reconstruction and causal traversal.
   - Escalation alert generation (`CorrelationAlert`) and external feedback signal ($E(t)$) dispatched to C1.

## Directory Structure

```
c3-cross-domain/
├── Data/
│   └── input/               # Raw Host and IoT flow CSV telemetry
├── Stages/
│   ├── Stage 2/             # Preprocessing & schema validation notebooks & artifacts
│   ├── Stage 3/             # Relationship mapping notebooks & link datasets
│   ├── Stage 3b/            # Scenario grouping & temporal windowing
│   ├── Stage 4/             # Graph construction, edge tables, feature matrices & splits
│   └── Stage 5_R-GCN/       # R-GCN architecture, checkpoints (epoch 1-40), validation & test logs
├── cache/                   # Reference embeddings cache
├── src/
│   └── xdr_c3/              # Core Python library
├── tests/                   # Pytest verification and validation test suite
├── verification/            # Historical comparison scripts and ground truth records
├── pytest.ini
└── requirements.txt
```

## Running Tests

From the `c3-cross-domain` directory:

```bash
pytest
```
