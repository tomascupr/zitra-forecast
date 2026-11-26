---
pipeline_tag: time-series-forecasting
library_name: zitra-forecast
tags:
- time-series
- forecasting
- probabilistic-forecasting
- foundation-model
- timestral
- safetensors
- pytorch
model-index:
- name: Zitra-209M-v2
  results:
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: fev-bench
      type: autogluon/fev-bench
    metrics:
    - name: SQL skill score
      type: skill-score
      value: 50.45
    - name: WQL skill score
      type: skill-score
      value: 54.81
    - name: MASE skill score
      type: skill-score
      value: 39.56
    source:
      name: fev-bench leaderboard
      url: https://huggingface.co/spaces/autogluon/fev-bench
  - task:
      type: time-series-forecasting
      name: Time Series Forecasting
    dataset:
      name: GIFT-Eval
      type: Salesforce/GiftEval
    metrics:
    - name: MASE
      type: mase
      value: 0.6578
    - name: CRPS
      type: crps
      value: 0.4451
    source:
      name: GIFT-Eval leaderboard
      url: https://huggingface.co/spaces/Salesforce/GIFT-Eval
license: other
---

# ⚡ Zitra-209M-v2

**Zitra-209M-v2** is an exact **`209,312,256`-parameter, 1.00× latency drop-in upgrade** to `mistralai/Timestral-209M` — delivering sharper probabilistic calibration across **GIFT-Eval** (`−0.20%` overall CRPS/WQL, `−0.23%` ex-solar) and **fev-bench** (`−0.41%` equal-task WQL, **`−0.74%` covariate WQL**, **`−0.67%` multivariate WQL**), plus **`−4.90% MASE / −5.65% WQL`** on live prospective count forecasting — with **zero extra parameters and zero latency overhead (`25.98 ms` vs `25.92 ms` on B300)**.

**Zitra v2 Family:** [**Zitra-Titan-585M** (Flagship Adaptive)](https://huggingface.co/tomcupr/Zitra-Titan-585M) · [**Zitra-Apex-585M** (Fast Covariate Specialist)](https://huggingface.co/tomcupr/Zitra-Apex-585M) · [**Zitra-209M-v2** (1.00× Speed Drop-In)](https://huggingface.co/tomcupr/Zitra-209M-v2) · [**Interactive Live Demo Space**](https://huggingface.co/spaces/tomcupr/zitra-v2-demo)

---

## Engineering Highlights

- **Algebraically Folded 24-Layer FullLoRA Model Soup:** Instead of tacking on runtime adapters that slow down serving, we post-trained 5 independent seeds of full-depth 24-layer rank-16 LoRA (`wQKV`, `wO`, `mlp.w3`) on clean non-benchmark real datasets plus regime-diverse synthetic trajectories, averaged the resulting updates into a variance-reduced model soup, and **algebraically folded all 420 LoRA matrices directly back into exact stock `TimestralModel` weights** (verified numerical parity `max_abs_diff = 2.15e-6`).
- **Exact `1.003×` Latency Drop-In:** Identical architecture, identical VRAM footprint, identical `config.json`, and identical serving throughput to `Timestral-209M` — plus exact nonnegative integer-support rounding when a series exhibits strictly nonnegative integer history over the last 84 steps.
- **Zero Benchmark Overlap:** Zero GIFT-Eval or fev-bench datasets touched during post-training or selection.

---

## Benchmarks vs Stock Timestral-209M

| Benchmark / Slice | **Zitra-209M-v2** | Stock Timestral-209M | Delta |
|---|---:|---:|---:|
| **B300 Warm Latency** `(1,1,672)→72` | **25.98 ms** | 25.92 ms | **1.003× (zero overhead)** |
| **fev-bench Native Skill** (MASE / SQL / WQL) ↑ | **39.56 / 50.45 / 54.81** | 39.49 / 50.39 / 54.63 | **Wins all 3** |
| **fev-bench Equal-Task (100 tasks)** ↓ | `−0.12% / −0.41% / −0.12%` | baseline | **52/100 tasks both lower** |
| **fev-bench Covariates (42 tasks)** ↓ | **`−0.32% MASE / −0.74% WQL / −0.40% SQL`** | baseline | Strongest single-backbone lift |
| **fev-bench Multivariate (35 tasks)** ↓ | **`−0.19% MASE / −0.67% WQL`** | baseline | Improved cross-variate calibration |
| **GIFT-Eval CRPS / WQL** (97 / Ex-Solar 89) ↓ | **0.4451 / 0.4461** | 0.4460 / 0.4471 | **`−0.20%` / `−0.23%`** |
| **Internal 2026 Rohlik Holdout** (8 tasks, 10,859 forecasts) ↓ | **`−0.07% MASE / −0.18% WQL / −0.16% WAPE`** | baseline | **Wins 6/8 tasks** (`−0.78%` WAPE on 5,000 SKUs) |
| **Live Prospective Bikeshare (`H=64` = 128m)** ↓ | **`−4.90% MASE / −5.65% WQL`** | baseline | Beats stock in both cities |

### Choose Your Zitra v2 Tier

| Model | Params | B300 Latency | Best For |
|---|---:|---:|---|
| **[Zitra-Titan-585M](https://huggingface.co/tomcupr/Zitra-Titan-585M)** | 584.4M | 251 ms (`9.7×`) | **Maximum overall accuracy (`0.6443 / 0.4352` GIFT-Eval)** |
| **[Zitra-Apex-585M](https://huggingface.co/tomcupr/Zitra-Apex-585M)** | 584.5M | 84 ms (`3.2×`) | **Fast 3-prior retail & covariate forecasting** |
| **[Zitra-209M-v2](https://huggingface.co/tomcupr/Zitra-209M-v2)** | 209.3M | 26 ms (`1.0×`) | **Zero-cost drop-in upgrade to Timestral-209M** |

---

## Quickstart

```bash
pip install "zitra-forecast @ git+https://github.com/tomascupr/zitra-forecast.git"
```

```python
import numpy as np
from zitra import ZitraForecaster

# Exact drop-in replacement for Timestral-209M (same speed, same VRAM)
model = ZitraForecaster.from_pretrained("tomcupr/Zitra-209M-v2")

context = np.random.default_rng(42).standard_normal(1000).cumsum()
fc = model.forecast(context, prediction_length=96)

median = fc.median
lower, upper = fc.quantile(0.1), fc.quantile(0.9)
```

*(You can also load `model.safetensors` + `config.json` directly with `TimestralForecaster.from_pretrained("tomcupr/Zitra-209M-v2")` without installing `t0` or `chronos`!)*

---

## License

Derived from [`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M) and released under the **Modified MIT License**. Commercial use is subject to the upstream Timestral Modified MIT terms.
