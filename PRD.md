# PRD: Predictive and Confidence-Aware ML for Proactive WiFi AP Load Balancing
**Course:** Wireless Networks · IIITDM Kancheepuram  
**Team:** Cupcake (Team 5) — CS23B2043 · CS23B1085 · CS23B1004 · CS23I1041  
**Document type:** Project Requirements Document (PRD) · v4.0  
**Deliverable format:** Kaggle-compatible Jupyter Notebooks + this document

---

## 1. Project Overview

This project implements and compares **three ML approaches** to proactive WiFi Access Point (AP) load balancing, directly addressing the five research gaps identified in our literature survey. The core innovation is a shift from *reactive* AP selection (responding to current congestion) to *predictive* load management with **time-to-congestion estimation** and **confidence-aware decision gating**.

All modeling and experimentation live in Kaggle notebooks. The final output is a comparative analysis of the three approaches, demonstrating which architecture best addresses the identified gaps under academic scope constraints.

---

## 2. Problem Statement

Traditional WiFi load balancing triggers a handover *after* an AP becomes congested. This causes:
- Hotspot formation (some APs overloaded, others idle)
- Reactive handovers with QoS degradation during transition
- No uncertainty quantification → harmful low-confidence handover decisions

**We frame the core ML task as:**

> Given a time window of AP telemetry features (connected clients, channel utilization, temporal context, and neighbor load), **predict the time remaining (in intervals) until each AP crosses a defined congestion threshold**, and attach a **confidence score** to that prediction to gate handover decisions.

This is a **time-series regression + uncertainty quantification** problem, not classification.

---

## 3. Research Gaps Being Addressed

| Gap ID | Gap Description | How This Project Addresses It |
|--------|----------------|-------------------------------|
| Gap 1 | No proactive/predictive load balancing | Core task is TTC (time-to-congestion) regression |
| Gap 2 | No estimation of *time until* congestion | Explicit TTC regression label engineered from data |
| Gap 3 | No confidence-aware decision making | MC Dropout / Conformal Prediction in all three models |
| Gap 4 | Limited spatial-temporal context | `neighbor_avg_load` feature + GNN-lite in Notebook 3 |
| Gap 5 | No end-to-end pipeline demonstrated | Notebook 4 simulates the full controller loop |

---

## 4. Dataset Strategy

### 4.1 Overview: Two Separate Datasets — One Training Target

Notebook 0 produces **two separate, non-merged datasets**:

- **CRAWDAD dataset** — real-world syslog traces from Dartmouth campus. Processed and saved with a `crawdad_` prefix. Used for **distributional validation only** — not fed to any model.
- **Synthetic dataset** — CRAWDAD-informed generative model producing ~2.5 million rows. This is the **sole training dataset** for all model notebooks (NB01–NB05).

> **The real and synthetic data are never merged.** All models train, validate, and test exclusively on synthetic data. CRAWDAD data serves as a distributional reference anchor.

The trained dataset is published to Kaggle at:
```
/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/train.parquet
/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/val.parquet
/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/test.parquet
```

---

### 4.2 Real Data: CRAWDAD Dartmouth (Reference Only)

Three consecutive `.merged` syslog files (~2–3 days) are parsed:

```
/kaggle/input/datasets/sohanuuu/crawdad-syslog/060926.072827.merged
/kaggle/input/datasets/sohanuuu/crawdad-syslog/060927.080204.merged
/kaggle/input/datasets/sohanuuu/crawdad-syslog/060928.072608.merged
```

Raw fields extracted per event: `timestamp`, `ap_id` (bssid), `client_id` (MAC), `event_type` (up/down). Client roaming is handled by evicting a client from its previous AP before registering on the new one. Events are aggregated into **30-second windows** per AP to compute `clients_connected` and `clients_delta`.

This real data is used to:
- Learn statistical parameters for the synthetic generator (load distributions, temporal trends)
- Serve as a distributional reference in Notebook 0 EDA (KS test, overlay plots)
- Be saved as `crawdad_train/val/test.parquet` for Notebook 5 audit

**CRAWDAD data is not fed to any model.**

---

### 4.3 Synthetic Data: CRAWDAD-Informed Generative Model

Synthetic data is generated using a physically-motivated model that learns patterns from CRAWDAD and scales to ~2.5 million rows across hundreds of synthetic APs.

#### AP Personality Types

Each synthetic AP is assigned a personality that governs its baseline load, noise scale, and burst behavior:

| Type | Fraction | Baseline Clients | Noise Std | Burst Frequency |
|------|----------|-----------------|-----------|-----------------|
| idle | 30% | 0 | 0.3 | 1–2 / day |
| moderate | 50% | 1–4 | 1.0 | 3–5 / day |
| busy | 20% | 5–15 | 2.0 | 4–6 / day |

#### Signal Components

Each AP's load trace is composed of additive components:

```
clients(t) = base_trend(t) + AR(1)_noise(t) + burst_signal(t)
```

- **Daily periodicity** — sinusoidal multiplier peaking mid-morning to simulate campus usage
- **AR(1) correlated noise** — mean-reverting residuals (coefficient 0.6–0.9) for session persistence
- **Clustered burst events** — 70% of bursts biased toward campus peak hours (8–10am, 12–2pm, 5–7pm); rise + exponential decay shape; not isolated spikes

#### TTC Calibration

After generation, a post-processing pass ensures **≥30% of synthetic APs have at least one congestion crossing** (load ≥ 35). APs below this threshold receive a targeted burst injection at their local load peak. This guarantees sufficient near-zero TTC labels for model training.

#### Validation Criteria

The generative model is verified against five calibration assertions before the dataset is saved:

| Check | Target |
|-------|--------|
| TTC mean | [8, 16] intervals |
| Zero-inflation fraction | [45%, 65%] |
| 95th percentile load | [10, 25] clients |
| KS statistic vs. CRAWDAD | < 0.05 |
| APs with congestion events | ≥ 30% |

---

### 4.4 Final Dataset Schema

After feature engineering, both datasets share this schema. **Only the synthetic splits are consumed by model notebooks.**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | 30-second window start time |
| `ap_id` | string | BSSID (real) or `<bssid>_synth_v<n>` (synthetic) |
| `clients_connected` | int | Associated clients at time t |
| `clients_delta` | int | Change vs. previous window |
| `source` | string | `real` or `synthetic` |
| `hour_of_day` | int | Hour from timestamp (0–23) |
| `day_of_week` | int | Day from timestamp (0–6) |
| `channel_utilization` | float | `clients_connected / 61.0`, clipped to [0, 1] |
| `neighbor_avg_load` | float | Mean `clients_connected` across all APs at same timestamp |
| `ttc` | float | Steps until load ≥ 35 within 60-step horizon; `NaN` if never congested |

**Sample rows from the actual Notebook 0 output:**

| timestamp | ap_id | clients_connected | clients_delta | source | hour_of_day | day_of_week | channel_utilization | neighbor_avg_load |
|-----------|-------|-------------------|---------------|--------|-------------|-------------|---------------------|-------------------|
| 2006-09-26 14:53:30 | 00:0b:86:00:66:50 | 1 | 1 | real | 14 | 1 | 0.016393 | 1.790829 |
| 2006-09-26 14:54:00 | 00:0b:86:00:66:50 | 0 | -1 | real | 14 | 1 | 0.000000 | 1.793242 |
| 2006-09-26 14:54:30 | 00:0b:86:00:66:50 | 0 | 0 | real | 14 | 1 | 0.000000 | 1.799678 |

*(The synthetic parquet files consumed by model notebooks have the same schema with `source = synthetic` and populated `ttc` values.)*

---

### 4.5 Feature Engineering Rationale

| Feature | Category | Why It's Included |
|---------|----------|-------------------|
| `hour_of_day`, `day_of_week` | Temporal | Captures periodic campus load patterns |
| `clients_delta` | Dynamic behavior | Rate of change is more predictive of imminent congestion than absolute count alone |
| `neighbor_avg_load` | Spatial context | Cross-AP load awareness; partially addresses Gap 4 for all three models |
| `channel_utilization` | Utilization metric | Normalized RF occupancy proxy: `clients / 61.0` |
| `ttc` | **Regression target** | Steps until congestion crossing; `NaN` rows are dropped before model training |

---

### 4.6 Dataset Split Strategy

Splits are **strictly chronological** (no shuffle) applied independently to each dataset.

| Split | Fraction | Usage |
|-------|----------|-------|
| Train | 70% (earliest) | Model training |
| Val | 15% (middle) | Hyperparameter tuning, early stopping |
| Test | 15% (latest) | Final evaluation |

Split integrity asserted in code:
```python
assert train_df['timestamp'].max() < val_df['timestamp'].min()
assert val_df['timestamp'].max() < test_df['timestamp'].min()
```

---

### 4.7 Key Dataset Decision

> **All model notebooks train exclusively on the synthetic dataset. CRAWDAD data is a generative reference and distributional audit baseline — not a training input.**

This choice is made because:
- CRAWDAD traces have insufficient congestion events for TTC label engineering at scale
- The synthetic generator produces a calibrated, assertion-verified dataset (~2.5M rows) with ≥30% of APs having congestion crossings
- The `source` column and CRAWDAD parquet files are retained so Notebook 5 can run a distributional audit as a data quality check

---

## 5. Notebooks Strategy

### Overview

The project is organized into **six sequential notebooks**. Notebooks 1–3 all read from the same Kaggle-mounted synthetic dataset and are independently runnable. Notebook 4 depends on at least one saved model checkpoint. Notebook 5 aggregates all metrics files.

All notebooks run on **Kaggle (Python kernel, GPU T4 available for Notebook 1)**.

---

### Notebook 0: `00_data_preparation.ipynb`
**Title:** Data Ingestion, Synthetic Dataset Construction & Label Engineering  
**Purpose:** Produces the shared processed dataset consumed by all model notebooks  
**Output:** Published to Kaggle as `sohanuuu/synthetic-from-crawdad`

**What this notebook does (verified against actual code):**

1. **Parse CRAWDAD syslogs** — regex-based event extraction with cross-AP roaming handling (evict client from old AP before registering on new one)
2. **30-second aggregation** — resample events to fixed windows per AP; `clients_delta` = net event sum per window
3. **Generate ~2.5M synthetic rows** — AP personalities (idle/moderate/busy), daily periodicity, AR(1) noise, clustered peak-hour bursts
4. **TTC calibration pass** — boost APs below congestion crossing count to meet ≥30% target
5. **Stamp `source` column before feature engineering** — `real` for CRAWDAD, `synthetic` for generated rows
6. **Feature engineering (applied independently to each dataset):**
   - `hour_of_day`, `day_of_week` from timestamp
   - `channel_utilization = clients_connected / 61.0`
   - `neighbor_avg_load` = cross-AP mean at each timestamp
7. **Compute TTC independently per dataset** — per AP, `NaN` where 60-step horizon not reached
8. **Data quality validation** — KS test, distribution plots, AP type breakdown, 5 calibration assertions
9. **Chronological 70/15/15 split applied separately to each dataset**
10. **Save outputs:**
    - `crawdad_train/val/test.parquet` → real data for reference
    - `processed/train.parquet`, `processed/val.parquet`, `processed/test.parquet` → **synthetic only**, published to Kaggle

> **The `processed/` parquet files contain only synthetic rows.** This is what all downstream model notebooks consume.

---

### Notebook 1: `01_model_tcn_mc_dropout.ipynb`
**Title:** Approach A — Temporal Convolutional Network (TCN) with Monte Carlo Dropout  
**Innovation addressed:** Gap 1, 2, 3  
**Dataset input:** `/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/{train,val,test}.parquet`

**Feature alignment with synthetic schema:** All 8 model input features (`clients_connected`, `clients_delta`, `hour_of_day`, `day_of_week`, `channel_utilization`, `neighbor_avg_load`, + 2 lag-derived) are present in the synthetic parquet. `ttc` is the regression target; rows where `ttc = NaN` are dropped before training.

**Architecture:**
```
Input: [batch, seq_len=10, features=8]
  → Conv1D(filters=64, kernel=3, dilation=1) → ReLU
  → Conv1D(filters=64, kernel=3, dilation=2) → ReLU   ← 20-step receptive field
  → Conv1D(filters=64, kernel=3, dilation=4) → ReLU   ← 40-step receptive field
  → Dropout(p=0.3)   ← kept ACTIVE at inference for MC
  → GlobalAvgPool
  → Dense(1)         → TTC prediction
```

**Uncertainty via MC Dropout:** T=50 stochastic forward passes at inference. Mean = point estimate; Std = uncertainty proxy. Confidence = `1 / (1 + std)`.

**Training:** Huber loss, Adam + cosine annealing, early stopping on val MAE.

**Sections:**
1. Load synthetic train/val/test parquets; drop `ttc = NaN` rows
2. Sliding window sequence construction (seq_len=10)
3. TCN model + MC Dropout wrapper
4. Training loop
5. MC Dropout inference + uncertainty extraction
6. Calibration: confidence vs. actual error
7. Metrics: MAE, RMSE, prediction interval coverage
8. Visualization: prediction vs. ground truth with uncertainty bands
9. Save `model_a.pt` + `metrics_a.json`

---

### Notebook 2: `02_model_xgboost_conformal.ipynb`
**Title:** Approach B — Gradient Boosting (XGBoost) with Conformal Prediction  
**Innovation addressed:** Gap 1, 2, 3  
**Dataset input:** `/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/{train,val,test}.parquet`

**Feature alignment with synthetic schema:** Sequences are flattened into lag feature vectors (t, t-1, ..., t-9) across all columns in the synthetic schema. No sequence construction needed — rows become 1D feature vectors. `ttc = NaN` rows are dropped.

**Conformal Prediction Setup:** Split conformal using a calibration hold-out. Prediction interval `[y_hat − q, y_hat + q]` with guaranteed 90% marginal coverage. Conformity score: `|y_true − y_hat|`.

**Sections:**
1. Load synthetic parquets; flatten into lag vectors; drop NaN TTC
2. XGBoost training with Optuna HPO (50 trials, minimize val MAE)
3. Conformal Prediction calibration
4. Test-time inference: point prediction + interval
5. SHAP beeswarm (top 10 features)
6. Metrics: MAE, RMSE, interval width, empirical coverage @ 90%
7. Save `metrics_b.json`

---

### Notebook 3: `03_model_gnnlite_xgb_ensemble.ipynb`
**Title:** Approach C — GNN-lite Graph Feature Enrichment + XGBoost Ensemble  
**Innovation addressed:** Gap 1, 2, 3, **4**  
**Dataset input:** `/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/{train,val,test}.parquet`

**Feature alignment with synthetic schema:** The synthetic schema includes `neighbor_avg_load` as a pre-computed scalar. Notebook 3 extends this by constructing a full k=4 AP neighbor graph from coordinate metadata and computing per-neighbor feature vectors — a strict superset of what Notebooks 1 and 2 use.

**Architecture:**

Step 1 — AP neighbor graph (k=4 nearest, Euclidean on campus grid)

Step 2 — One-hop message passing:
```python
neighbor_features = mean(features of k=4 nearest neighbors at t)
enriched_features = concat(own_features, neighbor_features)
```

Step 3 — XGBoost ensemble (seeds 42, 123, 999):
```python
# Point estimate = mean(3 models); Uncertainty = std(3 models)
```

**Sections:**
1. Load synthetic parquets + AP coordinate metadata
2. One-hop message passing: enriched features for all APs × timestamps
3. XGBoost ensemble training (seeds 42, 123, 999)
4. Ensemble prediction (mean + std)
5. Ablation: MAE with vs. without neighbor enrichment (quantifies Gap 4)
6. SHAP beeswarm on enriched feature set
7. Metrics + save `metrics_c.json`

---

### Notebook 4: `04_controller_simulation.ipynb`
**Title:** End-to-End Proactive Controller Simulation (Gap 5)  
**Dataset input:** `/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/test.parquet`

**This notebook does NOT train a model.** It loads the best checkpoint from Notebooks 1–3 and replays the test set as a simulated real-time controller.

**Controller Logic:**
```python
def controller_step(ap_states, model, confidence_threshold=0.7, ttc_threshold=120):
    ttc_preds, confidences = model.predict_with_uncertainty(ap_states)
    for ap in ap_states:
        if ttc_preds[ap] < ttc_threshold and confidences[ap] > confidence_threshold:
            target_ap = select_least_loaded_neighbor(ap, ap_states)
            trigger_handover(ap, target_ap)
        elif confidences[ap] <= confidence_threshold:
            increase_monitoring_frequency(ap)
```

**Simulation scenarios:** (1) RSSI-threshold reactive baseline, (2) Approach A with TCN + MC Dropout gating, (3) best-performing approach from Notebooks 1–3.

**Metrics:** congestion event rate, handover efficiency, false positive rate, QoS proxy.

---

### Notebook 5: `05_comparison_report.ipynb`
**Title:** Comparative Analysis & Results  
**Inputs:** `metrics_a/b/c.json` + `crawdad_test.parquet` for distributional audit

**Sections:**
1. Load all metrics JSONs
2. Unified comparison table across Approaches A, B, C
3. **Distribution audit** — compare synthetic training distribution against CRAWDAD reference using KS test and load distribution overlay. Since no model sees CRAWDAD data, this section quantifies how well the synthetic generator captured real-world patterns.
4. Error analysis by scenario (rush hour, off-peak, burst spike)
5. Research gap coverage matrix
6. Limitations and future work

---

### Notebook Dependency Map

```
Notebook 0 ── generates ──→  /kaggle/.../synthetic-from-crawdad/
                               train.parquet  val.parquet  test.parquet
                                         │
             ┌───────────────────────────┼──────────────────────────┐
             ↓                           ↓                          ↓
       Notebook 1                  Notebook 2                 Notebook 3
    TCN + MC Dropout            XGBoost + Conformal       GNN-lite + Ensemble
    → model_a.pt                 → metrics_b.json          → metrics_c.json
    → metrics_a.json
             │                           │                          │
             └───────────────────────────┼──────────────────────────┘
                                         ↓
                                   Notebook 4
                               Controller Simulation
                               ← loads best checkpoint
                                         ↓
                                   Notebook 5
                               Comparison Report
                               ← loads metrics_a/b/c.json
```

Notebooks 1–3 have no dependency on each other and can run in parallel.

---

## 6. Three Models: Design Rationale Summary

| | Approach A | Approach B | Approach C |
|--|------------|------------|------------|
| **Model** | TCN | XGBoost | GNN-lite + XGBoost Ensemble |
| **Uncertainty** | MC Dropout | Conformal Prediction | Seed Ensemble (std across 3 seeds) |
| **Temporal** | ✓ (dilated conv, receptive field 40 steps) | ✓ (lag features t to t-9) | ✓ (lag features on enriched input) |
| **Spatial** | Partial (`neighbor_avg_load` scalar) | Partial (`neighbor_avg_load` scalar) | ✓ (full one-hop neighbor aggregation) |
| **Training data** | Synthetic only | Synthetic only | Synthetic only |
| **Gaps covered** | 1,2,3 | 1,2,3 | 1,2,3,4 |
| **Interpretability** | Low–Medium | High (SHAP) | High (SHAP on enriched features) |

---

## 7. Technical Stack (Kaggle Environment)

```python
# Core
numpy, pandas, scikit-learn, matplotlib, seaborn, scipy

# Deep learning (pre-installed on Kaggle)
torch >= 2.0          # TCN (Conv1D + MC Dropout)
torchmetrics          # MAE, RMSE

# Gradient Boosting
xgboost >= 2.0
optuna                # hyperparameter search (Notebook 2)
shap                  # feature importance (Notebooks 2 & 3)

# Uncertainty
crepes                # Conformal Prediction (pip install in notebook)
# OR: mapie

# Utilities
pyarrow               # parquet I/O
tqdm                  # progress bars
```

---

## 8. Agent Definitions

### Agent 1: `@data-engineer`
**Owns:** Notebook 0  
**Predefined prompts:**
- "Parse CRAWDAD `.merged` syslogs with roaming-aware client tracking"
- "Generate ~2.5M synthetic rows: AP personality model + AR(1) noise + clustered bursts"
- "Run TTC calibration: ensure ≥30% of synthetic APs have congestion crossings"
- "Verify 5 calibration assertions before saving parquets"
- "Save synthetic splits to `processed/` (generic names) for downstream notebooks; save CRAWDAD splits with `crawdad_` prefix"

### Agent 2: `@tcn-modeler`
**Owns:** Notebook 1  
**Predefined prompts:**
- "Load `/kaggle/input/datasets/sohanuuu/synthetic-from-crawdad/train.parquet`; drop NaN TTC rows"
- "Implement TCN with dilation rates [1, 2, 4], GlobalAvgPool, Huber loss"
- "Keep dropout active at inference for MC Dropout with T=50 forward passes"
- "Save `model_a.pt` and `metrics_a.json`"

### Agent 3: `@xgboost-uncertainty`
**Owns:** Notebook 2  
**Predefined prompts:**
- "Load synthetic parquets; flatten into lag vectors (t to t-9); drop NaN TTC"
- "Run Optuna 50-trial study for XGBoost val MAE"
- "Implement split conformal prediction with 90% coverage guarantee"
- "Generate SHAP beeswarm for top-10 synthetic schema features"

### Agent 4: `@gnnlite-spatial`
**Owns:** Notebook 3  
**Predefined prompts:**
- "Build k=4 AP neighbor graph from coordinate metadata"
- "One-hop message passing: concat own features with mean of k=4 neighbors"
- "Train XGBoost ensemble (seeds 42, 123, 999) on enriched features"
- "Ablation: MAE with vs. without neighbor enrichment"

### Agent 5: `@controller-sim`
**Owns:** Notebook 4  
**Predefined prompts:**
- "Load synthetic test parquet and best model checkpoint"
- "Simulate reactive baseline and confidence-gated proactive controller"
- "Count congestion events, handovers, false positives"

### Agent 6: `@report-writer`
**Owns:** Notebook 5  
**Predefined prompts:**
- "Load all `metrics_*.json` and render unified comparison table"
- "Run KS test and distribution overlay: synthetic training data vs. CRAWDAD reference"
- "Generate research gap coverage matrix"
- "Export comparison table as LaTeX"

---

## 9. Evaluation Metrics

### Primary (TTC Regression Quality)
- **MAE** — mean absolute error in predicted TTC (intervals)
- **RMSE** — penalizes large errors more
- **R²** — variance explained

### Uncertainty Quality
- **Empirical coverage** — fraction of test samples where true TTC falls inside predicted interval (target: ≥90% for 90% intervals)
- **Mean interval width** — narrower is better at same coverage
- **Calibration curve** — reliability diagram (confidence bin vs. actual accuracy)

### Controller Performance (Notebook 4)
- **Congestion event rate** — events per hour exceeding threshold
- **Handover efficiency** — congestion events prevented per handover triggered
- **False positive rate** — handovers triggered when TTC was actually > horizon

### Data Quality Audit (Notebook 5)
- KS statistic: synthetic training distribution vs. CRAWDAD reference
- TTC distribution shape comparison
- AP personality mix verification (30/50/20 idle/moderate/busy)

---

## 10. Scope Constraints (Academic)

- No real hardware deployment or live network integration
- No federated learning
- No RL-based controller (regression + rule-based, not MDP)
- No LiFi/hybrid networks
- Models trained only on synthetic data; CRAWDAD is a reference baseline, not a training input
- Conformal Prediction uses split conformal only
- Notebooks 1 and 2 use only the pre-computed `neighbor_avg_load` scalar, not a full neighbor feature vector (that is Notebook 3's contribution)

---

## 11. Timeline

| Week | Milestone |
|------|-----------|
| Week 1 | CRAWDAD access + Notebook 0 complete; synthetic dataset published to Kaggle |
| Week 2 | Notebook 1 (TCN + MC Dropout) + Notebook 2 (XGBoost + Conformal) complete |
| Week 3 | Notebook 3 (GNN-lite + Ensemble) + Notebook 4 (Controller Simulation) complete |
| Week 4 | Notebook 5 (Comparison + distribution audit) + course report writeup |

---

## 12. Key Design Decisions & Justifications

**Why train only on synthetic data, not CRAWDAD?**  
CRAWDAD traces have very few congestion events — load rarely crosses the threshold in 2–3 days of campus data. Most rows would have `ttc = NaN` and be dropped, leaving an insufficient and severely imbalanced training set. The synthetic generator resolves this by calibrating ≥30% of APs to have congestion crossings, providing sufficient positive TTC labels.

**Why keep CRAWDAD at all if it is not used for training?**  
CRAWDAD provides the statistical substrate for the generative model (load distributions, temporal shape). It also serves as the external anchor for Notebook 5's distributional audit — comparing synthetic model-ready data against real-world patterns is the primary data quality check.

**Why are the datasets not merged?**  
Merging would contaminate the training set with CRAWDAD rows that have almost no congestion events, re-introducing the TTC label imbalance problem and invalidating the calibration assertions built into the synthetic generator.

**Why TTC regression instead of binary congestion classification?**  
Binary classification only tells you *if* congestion will occur, not *when*. To trigger a proactive handover with appropriate lead time (Gap 2), a time estimate is required.

**Why confidence gating instead of always-on handovers?**  
All four surveyed papers trigger actions without uncertainty awareness. A model uncertain about its prediction should not initiate a disruptive handover. Confidence gating (Gap 3) prevents this.

**Why TCN over LSTM for Approach A?**  
Fully parallelisable, 2–3× faster on Kaggle T4. Dilated convolutions at rates [1, 2, 4] model temporal patterns at 10, 20, and 40 steps without vanishing gradient issues.

**Why GNN-lite + XGBoost Ensemble over Transformer + Deep Ensemble for Approach C?**  
The Transformer required positional encoding, multi-head attention, learned AP embeddings, and 3 full deep training runs — high risk in a 5-day window. GNN-lite delivers the same spatial contribution (Gap 4) via one-hop aggregation, and the 3-seed XGBoost ensemble gives uncertainty with trivial overhead.

---

## 13. References

1. CRAWDAD dartmouth/campus — https://ieee-dataport.org/open-access/crawdad-dartmouthcampus-v-2004-11-09
2. CRAWDAD kth/campus — https://ieee-dataport.org/open-access/crawdad-kthcampus
3. Shattal et al. (2018) — ML-aided load balancing in dense IoT networks. *Sensors* 18(11):3779
4. Sifaou et al. (2022) — A-TCNN for hybrid LiFi/WiFi. arXiv:2208.05035
5. Almasan et al. (2022) — ML-based handover prediction in WLAN. *JNSM* 30(4)
6. Li et al. (2024) — Distributed learning for WiFi AP load prediction. arXiv:2405.05140
7. Angelopoulos et al. (2023) — Conformal Risk Control. *ICLR 2023*
8. Gal & Ghahramani (2016) — Dropout as Bayesian Approximation. *ICML 2016*
9. Lakshminarayanan et al. (2017) — Deep Ensembles. *NeurIPS 2017*
10. Bai et al. (2018) — An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271
