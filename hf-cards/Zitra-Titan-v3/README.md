---
license: apache-2.0
pipeline_tag: time-series-forecasting
library_name: zitra-forecast
tags:
- time-series
- forecasting
- probabilistic-forecasting
- foundation-model
- multi-architecture
- multivariate
- covariates
- apache-2.0
- safetensors
- pytorch
model-index:
- name: Zitra-Titan-v3
  results:
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: GIFT-Eval
      type: Salesforce/GiftEval
    metrics:
    - name: MASE
      type: mase
      value: 0.6522
    - name: CRPS
      type: crps
      value: 0.4422
    source:
      name: GIFT-Eval leaderboard
      url: https://huggingface.co/spaces/Salesforce/GIFT-Eval
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: fev-bench
      type: autogluon/fev-bench
    metrics:
    - name: SQL skill score
      type: skill-score
      value: 50.18
    - name: WQL skill score
      type: skill-score
      value: 54.52
    - name: MASE skill score
      type: skill-score
      value: 39.05
    source:
      name: fev-bench leaderboard
      url: https://huggingface.co/spaces/autogluon/fev-bench
---

# ⚡ Zitra-Titan-v3 (100% Pure Apache-2.0)

**Zitra-Titan-v3** is the flagship release of **Zitra v3** — a **100% Pure Apache-2.0**, commercially unrestricted probabilistic time-series foundation model bundling **four orthogonal Apache-2.0 architectures** into a single compact **`2.12 GB` `bfloat16`** `safetensors` checkpoint (`3.46 GB` GPU VRAM, `122 ms` on B300, **0 learned router weights**, **zero benchmark overlap**).

- **#1 Pure Apache-2.0 on Official GIFT-Eval (97 / 97 configs):** **`0.6522 MASE / 0.4422 CRPS`** — beats non-commercial `Timestral-209M` (`0.6575 / 0.4460`, `−0.82% / −0.86%`), `t0-gamma` (`0.661 / 0.452`, `−1.33% / −2.17%`), `TimesFM-3` (`0.667 / 0.456`, `−2.22% / −3.03%`), and `Chronos-2` (`0.698 / 0.485`, `−6.56% / −8.82%`).
- **#1 Pure Apache-2.0 on Official `fev-bench` (100 / 100 tasks):** **`39.05% MASE / 50.18% SQL / 54.52% WQL`** — the **first and only Pure Apache-2.0 model past 50% SQL skill**, ranking **#6 overall across 55 models** on the official leaderboard ahead of non-commercial `Timestral-209M` (#7) on **both SQL win rate (`80.2%` vs `79.4%`) and MASE win rate (`79.4%` vs `78.7%`)**, and beating `t0-gamma` (`+1.06 pp MASE / +0.76 pp SQL`) and `TimesFM-3` (`+1.19 pp SQL`).
- **Built-in Public Holiday Covariates (`240+` countries & regional subdivisions):** Pass `country="CZ"`, `"DE-BY"`, `"US-CA"`, etc. to `predict_df()` — no manual holiday engineering required.

[**Live Interactive Demo Space**](https://huggingface.co/spaces/tomcupr/zitra-v3-demo) · [**GitHub SDK (`zitra-forecast`)**](https://github.com/tomascupr/zitra-forecast)

![Zitra v3 Benchmarks on GIFT-Eval and fev-bench](https://raw.githubusercontent.com/tomascupr/zitra-forecast/main/assets/pareto_gift_fev.png)

---

## Why Zitra-Titan-v3 Works

Instead of forcing every time series through one frozen inductive bias (or restricting commercial users under non-production licenses), **Zitra-Titan-v3** unifies **four genuinely orthogonal Apache-2.0 foundation architectures** in one `2.12 GB` file and adapts dynamically at inference time with **zero learned router weights**:

1. **Four Orthogonal Pure Apache-2.0 Backbones (`1,061,831,429` total params, `2.124 GB` `bf16` on disk):**
   - **`Zitra-t0-256M-Dyn32K` (`255.6M`, Patch-Decoder Transformer):** Full-depth algebraically-folded FullLoRA model soup over `t0-gamma` with Dynamic RoPE `scale_base` (`512 / 1536 / 2048`), natively extending known-covariate context to **`16,384`** steps with zero short-horizon degradation.
   - **`Zitra-Chronos2-120M` (`119.5M`, T5 Encoder-Only):** Covariate-trained algebraically-folded FullLoRA soup over `Chronos-2`.
   - **`Zitra-PatchTST-385M` (`384.6M`, 30-Block Conformer):** Long-history univariate specialist derived from IBM Granite `PatchTST-FM-r2` (automatically gated to `0.000` on known-covariate series).
   - **`iAmTime-302M` (`302.1M`, T5 Encoder-Decoder 4-Expert MoE):** Native `arcsinh`-scaled multivariate & covariate MoE expert with vendored Apache-2.0 inference code.
2. **Fused-Batch Causal Test-Time Pop-Quiz Routing (`122 ms` B300; `102 ms` `fast=True`):**
   - Evaluates all active experts on horizon-matched causal rolling backtests inside fused GPU passes (`[main, -main, bt0, bt1]`), combining outlier-proof median-centered log-losses with effective sample-size shrinkage `n_eff / (n_eff + κ)` and cross-cutoff agreement penalties into an unclamped Bayesian posterior `w ∈ Δ⁴`.
   - Automatically triggers independent **Causal Self-Gated Cross-Inference (`cross_learning="auto"`)** on short univariate panels only when cross-series attention strictly beats independent inference on causal backtests.
3. **Exact Piecewise-Linear Mixture-CDF Inversion:**
   - Constructs `F_mix(y) = Σ w_e F_e(y)` and inverts it at every decile — guaranteeing a bounded influence function against minority-expert outliers and calibrated uncertainty widening when experts disagree.

---

## Complete Verified Benchmarks

Zero GIFT-Eval or `fev-bench` datasets were used during post-training or model selection (`trained_on_benchmark_datasets = []`). Every number comes from this single frozen checkpoint (`frozen.json` SHA-256 `f61bde12f03830e9`).

| Benchmark | Metric | **Zitra-Titan-v3** *(Apache-2.0)* | Timestral-209M *(Non-Comm)* | t0-gamma *(Apache-2.0)* | TimesFM-3 *(Non-Comm)* | Toto-2.0-2.5B *(Apache-2.0)* | Chronos-2 *(Apache-2.0)* |
|---|---|---:|---:|---:|---:|---:|---:|
| **GIFT-Eval** *(97 configs)* | MASE ↓ | **`0.6522`** | `0.6575` | `0.661` | `0.667` | `0.696` | `0.698` |
| | CRPS / WQL ↓ | **`0.4422`** | `0.4460` | `0.452` | `0.456` | `0.476` | `0.485` |
| **fev-bench** *(100 tasks)* | SQL Skill ↑ | **`50.18%`** | `50.39%` | `49.42%` | `48.99%` | `44.42%` | `47.28%` |
| | WQL Skill ↑ | **`54.52%`** | `54.63%` | `53.40%` | — | — | — |
| | MASE Skill ↑ | **`39.05%`** | `39.49%` | `37.99%` | `37.82%` | `32.54%` | `35.50%` |

- **Official GIFT-Eval Breakdown (97 / 97, strictly causal zero-leakage):** `−0.82% MASE / −0.86% CRPS` vs `Timestral-209M` (`−1.61% / −1.34%` ex-Jena), with standout family wins on `solar` (`−26.3% / −24.3%`), `temperature_rain` (`−10.5% / −17.1%`), `m4_hourly` (`−9.0%` CRPS), `bizitobs_application` (`−2.7% / −7.5%`), `kdd_cup_2018` (`−4.5% / −5.3%`), `loop_seattle` (`−3.4% / −4.1%`), `hospital` (`−0.9% / −3.0%`), `hierarchical_sales`, `restaurant`, `car_parts`, and `m4_monthly/quarterly/yearly`. Full validated 97-row CSV included as [`gift_eval_results.csv`](gift_eval_results.csv).
- **Official `fev-bench` Breakdown (100 / 100, strictly per-window zero-leakage):** `51 / 100` SQL wins vs `Timestral-209M`, including `past_cov_only` **`−3.84%` SQL**, `epf_de` **`−8.37%`**, `favorita_stores_1W` **`−12.28%`**, `rohlik_sales_1W` **`−11.89%`**, `uk_covid_nation_1W` **`−22.61%`**, and `solar_1W` **`−12.92%`**. Full 100-row CSV included as [`fev_bench_results.csv`](fev_bench_results.csv).

### Real-World Enterprise Retail & Fulfillment Proof (Rohlik, 100% Zero-Shot)

![Zero-Shot Rohlik Retail and Fulfillment Benchmarks](https://raw.githubusercontent.com/tomascupr/zitra-forecast/main/assets/rohlik_benchmark.png)

- **Public Rohlik Kaggle (`fev-bench`):** Reduces WQL vs `Timestral-209M` by **`−4.46%` / `−5.48%`** on Rohlik Orders (`1D` / `1W`) and **`−7.07%` / `−15.35%`** on Rohlik SKU Sales (`1D` / `1W`).
- **Internal 2026 Rohlik `D+7` Holidays (12 FCs across CZ, DE, AT, HU, RO):** Built-in public holiday covariates cut overall `D1–D7` WAPE from **`4.618% → 3.877%`** (`WQL 3.906% → 3.084%`) and slash 31 holiday weeks from **`13.577% → 7.805%`** (**`−24.9%`** lower than `Timestral-209M`'s `10.393%`).
- **Internal 2026 Rohlik 8 Production Ops Tasks (10,859 forecasts, no holiday covariates):** Wins **`6 / 8`** tasks vs `Timestral-209M` (**`−0.52% MASE / −0.45% WQL / −0.92% WAPE`**), including **`−4.21%` WAPE** on 24h hourly FC orders and **`−5.10%` WAPE** on 7-day hourly FC orders.

---

## Quickstart

### Install

```bash
pip install "zitra-forecast @ git+https://github.com/tomascupr/zitra-forecast.git"
```

### 5-Line NumPy / PyTorch Forecast

```python
import numpy as np
from zitra import ZitraForecaster

model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Titan-v3")

context = np.random.default_rng(42).standard_normal(512).cumsum()
fc = model.forecast(context, prediction_length=96)
# Optional single-pass fast mode (~102-115 ms on B300):
# fc = model.forecast(context, prediction_length=96, fast=True)

median = fc.median                      # (1, 1, 96)
q10, q90 = fc.quantile(0.1), fc.quantile(0.9)
print("Live pop-quiz expert weights:", fc.routing_summary())
```

### Pandas DataFrame API with Built-In Public Holidays

```python
import pandas as pd
from zitra import ZitraForecaster

model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Titan-v3")

pred_df = model.predict_df(
    df,
    prediction_length=14,
    id_column="item_id",
    timestamp_column="timestamp",
    target="target",
    country="CZ",                      # Supports 240+ countries & subdivisions (e.g. "DE-BY", "US-CA")
    quantile_levels=[0.1, 0.5, 0.9],
)
```

---

## Systems & Inference Engineering

Shipping four heterogeneous foundation architectures in production usually means brittle microservice glue, `4+ GB` of fragmented weights, and `300+ ms` sequential latency. **Zitra-Titan-v3** solves this end-to-end inside a single `2.12 GB` checkpoint:

- **Fused-Batch Single-Pass GPU Execution (`2.6×` speedup: `321 ms → 122 ms`):** Running symmetric sign-averaging (`+x`, `-x`) plus two horizon-matched causal rolling backtests (`bt0`, `bt1`) across four architectures naively triggers `16` sequential forward passes (`321 ms` on B300). `Zitra-Titan-v3` packs `[main, -main, bt0, bt1]` into fused padded GPU batches per active expert — cutting full 4-expert Pop-Quiz latency to **`122.4 ms`** median on B300, with an opt-in single-pass **`fast=True`** mode (**`101.9–115.1 ms`**) for latency-critical serving.
- **One `2.12 GB` `bfloat16` File, Mixed-Precision Numerical Discipline (`3.46 GB` VRAM):** Compresses all `1,061,831,429` parameters across four distinct model families into a single `model.safetensors` (`2.124 GB` on disk; fits on any `6–8 GB` consumer GPU). At load time, `PatchTST-385M` (30-layer Conformer) hydrates to `fp32` parameters and executes under `torch.amp.autocast("cuda", bfloat16)`, preserving exact `+0.007%` numerical parity with full `fp32` while avoiding `bf16` Conformer weight drift.
- **Zero Runtime LoRA Overhead (Algebraic Weight Folding + Multi-Seed Souping):** Every post-trained LoRA adapter is algebraically folded into base linear projections (`W' = W + (α/r)BA`) and souped across seeds prior to packaging — delivering full post-training gains with **zero adapter dispatch overhead** at inference time.
- **Dynamic RoPE `scale_base` Extrapolation (`16,384`-Step Context, Zero Short-Horizon Drift):** `Zitra-t0-256M-Dyn32K` dynamically scales RoPE frequency base by active context length (`scale_base = 512` for `≤8K`, `1536` for `8K–16K`, `2048` for `>16K`), extending known-covariate horizons to `16,384` steps (nearly 2 years of hourly telemetry) without degrading short-context accuracy.
- **Strict Batch / Singleton Invariance:** Attention-mask isolation (`group_ids`) guarantees a series produces identical predictions whether inferred solo or packed inside a large batch (`max abs diff ≤ 2e-5`, `0.0036%` relative across all 4 experts).
- **Zero-Compute Regime Gating & Structural Guards:**
  - *Covariate Gate:* On known-covariate tasks (such as `country="CZ"` public holidays), univariate `PatchTST-385M` is gated to `0.000` and **skipped completely** — saving GPU compute while dedicating 100% weight to covariate-native experts.
  - *Causal Self-Gated Cross-Learning (`cross_learning="auto"`):* Enables cross-series attention on short univariate panels **only** when it strictly beats independent inference on causal rolling backtests, eliminating negative transfer by construction.
  - *Monotonic / Cumulative Tail Guard:* Automatically detects cumulative/monotonic series and damps `arcsinh` tail extrapolation before Mixture-CDF inversion.
- **Exact Piecewise-Linear Mixture-CDF Inversion:** Inverts `F_mix(y) = Σ w_e F_e(y)` directly across the 21-quantile grid (`0.01 … 0.99`) rather than averaging quantiles linearly — ensuring a bounded influence function against outlier experts and calibrated epistemic interval widening when architectures disagree.
- **Self-Contained Pure Apache-2.0 Packaging:** Vendors minimal Apache-2.0 `iAmTime` T5-MoE inference modules directly inside the checkpoint (`_iamtime/`) with zero non-commercial dependencies and zero custom CUDA extensions required.

---

## Hardware & Memory Profile

| Metric | Verified Value |
|---|---:|
| Checkpoint on disk (`model.safetensors`, `bfloat16`) | **`2.124 GB`** *(single file, 4 architectures)* |
| Total parameters / Learned router parameters | **`1,061,831,429`** / **`0`** |
| GPU VRAM (single series / standard inference) | **`3.46 GB`** *(fits any 6–8 GB consumer GPU)* |
| GPU VRAM (large batch / `16K` context) | `6–12 GB` |
| B300 Median Latency — Full 4-Expert Pop-Quiz (univariate) | **`122.4 ms`** *(`2.6×` faster than sequential `321 ms`)* |
| B300 Median Latency — Full Pop-Quiz (known covariates, `H=48`) | **`173.7 ms`** *(PatchTST skipped automatically)* |
| B300 Median Latency — Single-Pass `fast=True` | **`101.9 ms`** *(cov)* / **`115.1 ms`** *(univariate)* |
| Singleton vs Batched Invariance (`cross_learning=False`) | **`≤ 2e-5`** max abs diff (`0.0036%` rel) |
| Context Length | Up to **`16,384`** steps (known covariates) / `8,192` steps (default) |
| Outputs | Full 21-quantile grid (`0.01 … 0.99`), nonnegative clipping, integer-count snapping |

## License & Upstream Attribution

**Apache License, Version 2.0** — 100% commercially permissive. All four bundled foundation components are Apache-2.0:
- **`t0.*` (`Zitra-t0-256M-Dyn32K`):** Derived from [`theforecastingcompany/t0-gamma`](https://huggingface.co/theforecastingcompany/t0-gamma) (**Apache-2.0**).
- **`chronos.*` (`Zitra-Chronos2-120M`):** Derived from [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) (**Apache-2.0**).
- **`patchtst.*` (`Zitra-PatchTST-385M`):** Derived from [`ibm-granite/granite-timeseries-patchtst-fm-r2`](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) (**Apache-2.0**).
- **`iamtime.*` (`iAmTime-302M`):** [`anishsaha/iAmTime-base-forecast`](https://huggingface.co/anishsaha/iAmTime-base-forecast) (**Apache-2.0**, arXiv:2603.22586).

See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for full attribution details.
