# Component 2: Agentless IoT Network Telemetry Intrusion Detection Engine

**Project:** Lightweight Agent-Based & Agentless XDR Framework for Hybrid Enterprise-IoT Environments (**J26-CS-310**)  
**Component:** Component 2 (C2)  
**Researcher:** Oshanee Jayathilaka (`IT23442320` | Jayathilaka N.M.O.R)

---

## 1. Overview & Research Novelty

Traditional IoT Network Intrusion Detection Systems (NIDS) trained on modern benchmarks frequently suffer from **shortcut learning**: models latch onto static network identifiers (IP addresses, port numbers, DNS query patterns) that correlate strongly with attack traffic *within a specific capture environment*, but fail completely when deployed in unobserved networks or under adversarial evasion.

### Core Threat Model & Novelty
Rather than relying on post-hoc statistical feature selection or heuristic pruning, **Component 2 enforces an adversarial threat-model-driven hard exclusion of five controllable identifiers BEFORE model training**:

$$\text{Excluded Identifiers} = \{\text{src\_ip}, \text{dst\_ip}, \text{src\_port}, \text{dst\_port}, \text{dns\_query\_has\_ip\_pattern}\}$$

An attacker compromising an IoT device has unilateral control over network routing attributes and transport header fields. A classifier dependent on these fields measures attacker preferences rather than malicious traffic dynamics. By restricting the feature space strictly to **behavioural and statistical traffic properties**, the detector achieves mathematical invariance to identifier perturbation and generalises across heterogeneous network topologies.

---

## 2. Feature Set Formulations

| Feature Set | Features | Description |
| :--- | :--- | :--- |
| **Set A (Baseline)** | 26 features | **Identifier-Inclusive**: Includes all behavioural flow features plus the 5 adversarially controllable identifiers (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `dns_query_has_ip_pattern`). Represents standard benchmark implementations in existing literature. |
| **Set B (Proposed)** | 21 features | **Behavioural-Only**: Strictly excludes all 5 controllable identifiers. Contains flow durations, packet volumes, byte asymmetries, inter-arrival statistics, and application-layer flags. |
| **Common-11** | 11 features | Intersection schema mapped between TON_IoT and IoT-23 for cross-dataset transfer evaluation (T3). |

---

## 3. Experimental Evaluation Framework

The research is evaluated across a rigorous three-tier experimental protocol:

```
+--------------------------------------------------------------------------+
|                        Evaluation Protocol Matrix                        |
+--------------------------------------------------------------------------+
|  T1: Clean Holdout Evaluation                                            |
|      - Standard 70/15/15 stratified train/val/test split                 |
|      - Quantifies clean baseline accuracy and macro F1                   |
+--------------------------------------------------------------------------+
|  T2: Adversarial Robustness Assessment                                    |
|      - T2a: Random identifier permutation across attacker-controlled fields|
|      - T2b: Plausible tuple resampling from benign distribution          |
|      - Directly tests whether the model collapses when identifiers spoof |
+--------------------------------------------------------------------------+
|  T3: Cross-Dataset Domain Transfer                                       |
|      - Trained on TON_IoT (Common-11 schema)                             |
|      - Tested on zero-shot unseen IoT-23 capture logs                     |
|      - Measures real-world generalization across different IoT topologies|
+--------------------------------------------------------------------------+
```

### Empirical Results Summary

#### T1 Clean Holdout Performance
| Model | Feature Set | Accuracy | Precision | Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Set A (Baseline) | 99.39% | 0.9816 | 0.9796 | 0.9806 |
| **Decision Tree** | Set B (Proposed) | 97.61% | 0.9472 | 0.9472 | 0.9471 |
| **Random Forest** | Set A (Baseline) | 99.52% | 0.9798 | 0.9869 | 0.9832 |
| **Random Forest** | Set B (Proposed) | 97.70% | 0.9475 | 0.9543 | 0.9506 |
| **Gradient Boosting** | Set A (Baseline) | 99.29% | 0.9721 | 0.9823 | 0.9768 |
| **Gradient Boosting** | Set B (Proposed) | 97.28% | 0.9344 | 0.9450 | 0.9391 |
| **LightGBM** | Set A (Baseline) | 99.49% | 0.9828 | 0.9896 | 0.9861 |
| **LightGBM** | Set B (Proposed) | 92.37% | 0.8758 | 0.8912 | 0.8756 |

#### T2a Adversarial Identifier Permutation (Robustness Proof)
| Model | Feature Set | Clean Macro F1 (T1) | Permuted Macro F1 (T2a) | F1 Drop |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | Set A (Baseline) | 0.9806 | 0.2252 | **-0.7553 (-77.0%)** |
| **Decision Tree** | Set B (Proposed) | 0.9471 | 0.9471 | **0.0000 (0.0% Invariant)** |
| **LightGBM** | Set A (Baseline) | 0.9861 | 0.5380 | **-0.4481 (-45.4%)** |
| **LightGBM** | Set B (Proposed) | 0.8756 | 0.8756 | **0.0000 (0.0% Invariant)** |

> **Key Finding:** While Set A models appear marginally superior on clean holdout data, they catastrophically collapse when an adversary perturbs identifiers (up to 75.5% drop in Macro F1). Set B models exhibit **strict zero-drop invariance**, proving resistance to identifier spoofing.

---

## 4. Repository Structure

```
c2-iot/
├── data/
│   ├── raw/
│   │   ├── train_test_network.csv       # TON_IoT benchmark dataset
│   │   └── iot23_samples/               # IoT-23 raw labeled connection captures
│   └── processed/
│       ├── feature_sets.json            # Exact schema definitions for Set A & B
│       ├── train_test_network_processed.csv
│       ├── set_a_identifier_inclusive.csv
│       ├── set_b_behavioural_only.csv
│       ├── common11_features.csv        # Extracted 11-feature common schema
│       └── iot23_processed_full.csv     # Harmonised IoT-23 evaluation flows
├── models/
│   ├── dt_set_a.joblib / dt_set_b.joblib
│   ├── rf_set_a.joblib / rf_set_b.joblib
│   ├── gb_set_a.joblib / gb_set_b.joblib
│   └── lgbm_set_a.joblib / lgbm_set_b.joblib
├── notebooks/
│   ├── 01_explore_data.ipynb             # Exploratory Data Analysis (EDA)
│   ├── 02_preprocessing.ipynb            # Cleaning, encoding & feature set partitioning
│   ├── 03_iot23_preprocessing.ipynb      # IoT-23 log parsing & schema harmonization
│   ├── 04a_model_dt.ipynb                # Decision Tree training (Set A vs Set B)
│   ├── 04b_model_rf.ipynb                # Random Forest training (Set A vs Set B)
│   ├── 04c_model_gb.ipynb                # Gradient Boosting training (Set A vs Set B)
│   ├── 04d_model_lgbm.ipynb              # LightGBM training (Set A vs Set B)
│   ├── 04e_model_comparison.ipynb        # Multi-model benchmarking & latency analysis
│   ├── 05_t1_formal_report.ipynb         # Formal T1 clean holdout evaluation
│   ├── 06_t2a_identifier_permutation.ipynb # T2a adversarial perturbation experiment
│   ├── 07_t2b_plausible_resampling.ipynb   # T2b tuple resampling experiment
│   └── 08_t3_common11_cross_dataset.ipynb  # T3 cross-dataset transfer evaluation
├── results/
│   ├── confusion_matrices/              # Per-model confusion matrix plots
│   ├── all_confusion_matrices.png       # Consolidated confusion matrix grid
│   ├── model_comparison_barchart.png    # Visual benchmark comparison
│   ├── dt_results.csv / rf_results.csv
│   ├── gb_results.csv / lgbm_results.csv
│   ├── full_model_comparison.csv
│   ├── t1_clean_holdout_results.csv
│   ├── t2a_permutation_results.csv
│   └── t2b_tuple_resampling_results.csv
└── README.md                            # Component documentation
```

---

## 5. Execution & Reproduction

All notebooks are pre-configured to execute sequentially with self-contained dependencies:

```bash
# 1. Environment setup
pip install numpy pandas scikit-learn lightgbm joblib matplotlib seaborn

# 2. Sequential notebook execution
jupyter notebook notebooks/
```

Execute notebooks `01` through `08` to reproduce data preprocessing, model artifact serialization, and evaluation tables.
