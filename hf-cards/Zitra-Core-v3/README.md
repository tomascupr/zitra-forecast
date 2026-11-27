---
license: apache-2.0
pipeline_tag: time-series-forecasting
library_name: zitra-forecast
tags:
- time-series
- forecasting
- probabilistic-forecasting
- foundation-model
- distilled
- multivariate
- covariates
- apache-2.0
- safetensors
- pytorch
model-index:
- name: Zitra-Core-v3
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
      value: 0.6551
    - name: CRPS
      type: crps
      value: 0.4473
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
      value: 49.69
    - name: WQL skill score
      type: skill-score
      value: 53.76
    - name: MASE skill score
      type: skill-score
      value: 38.39
    source:
      name: fev-bench leaderboard
      url: https://huggingface.co/spaces/autogluon/fev-bench
---

# ⚡ Zitra-Core-v3

**#1 Compact (256M) Fully Open (Apache-2.0) Time-Series Foundation Model** — outperforming `t0-gamma`, `TimesFM-3`, `Chronos-2`, and `Toto-2.0-2.5B` across both **GIFT-Eval** and **`fev-bench`**, at **8.8× the throughput of `t0-gamma`** (**28 ms** GPU · **86 ms** CPU).

Distilled from our flagship [`Zitra-Titan-v3`](https://huggingface.co/tomcupr/Zitra-Titan-v3), `Zitra-Core-v3` also runs **4.8× faster than `TimesFM-3`**, **2.1× faster than `Timestral-209M`**, and **1.5× faster than `Chronos-2`** — in just **1.0 GB VRAM** with built-in public holidays for 240+ countries.

[**Live Demo Space**](https://huggingface.co/spaces/tomcupr/zitra-v3-demo) · [**Flagship `Zitra-Titan-v3`**](https://huggingface.co/tomcupr/Zitra-Titan-v3) · [**GitHub SDK (`zitra-forecast`)**](https://github.com/tomascupr/zitra-forecast) · [**GIFT-Eval CSV**](gift_eval_results.csv) · [**fev-bench CSV**](fev_bench_results.csv)

![Zitra-Core-v3 Benchmarks on GIFT-Eval and fev-bench](https://raw.githubusercontent.com/tomascupr/zitra-forecast/main/assets/zitra_core_v3_benchmarks.png)

---

## Quickstart

```bash
pip install "zitra-forecast @ git+https://github.com/tomascupr/zitra-forecast.git"
```

```python
import numpy as np
from zitra import ZitraForecaster

model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Core-v3")
ctx = np.random.default_rng(42).standard_normal(512).cumsum()
fc = model.forecast(ctx, prediction_length=96)  # 28 ms GPU · 86 ms CPU

median, q10, q90 = fc.median, fc.quantile(0.1), fc.quantile(0.9)
```

**Pandas DataFrames with built-in public holidays** (`"CZ"`, `"DE-BY"`, `"US-CA"`, etc. — 240+ countries & subdivisions):

```python
pred_df = model.predict_df(
    df,
    prediction_length=14,
    id_column="store_id",
    timestamp_column="timestamp",
    target="orders",
    country="CZ",
    quantile_levels=[0.1, 0.5, 0.9],
)
```

---

## Verified Benchmarks

Every score below comes from this exact frozen checkpoint with strictly causal evaluation and **0.0% benchmark data overlap**.

### Official `fev-bench` (higher skill ↑ · lower runtime ↓)

| # | Model | License | Win Rate ↑ | SQL ↑ | Runtime ↓ |
|---:|---|---|---:|---:|---:|
| 6 | [Zitra-Titan-v3](https://huggingface.co/tomcupr/Zitra-Titan-v3) | Apache-2.0 | 80.1% | 50.30% | 5,719 s *(3.2×)* |
| 7 | Timestral-209M | Non-Comm. | 78.7% | 50.39% | 3,827 s *(2.1×)* |
| **11** | **⚡ Zitra-Core-v3** | **Apache-2.0** | **75.4%** | **49.69%** | **1,807 s (1.0×)** |
| 13 | t0-gamma | Apache-2.0 | 72.9% | 49.42% | 15,862 s *(8.8×)* |
| 18 | TimesFM-3 | Non-Comm. | 66.8% | 48.99% | 8,673 s *(4.8×)* |
| 22 | iAmTime | Apache-2.0 | 60.5% | 46.78% | 599 s *(0.3×)* |
| 24 | Chronos-2 | Apache-2.0 | 56.9% | 47.28% | 2,738 s *(1.5×)* |
| 25 | Toto-2.0-2.5B | Apache-2.0 | 55.9% | 44.42% | 13,835 s *(7.7×)* |

Ranks **#11 / 56 models** across SQL (75.4%), MASE (73.5%), and WQL (73.8%), with **53.76% WQL skill** and **38.39% MASE skill**. Beats `t0-gamma` on **63 / 100** tasks at **8.8×** throughput. Only our own `Zitra-Titan-v3` ranks higher among Apache-2.0 models.

### Official GIFT-Eval (lower is better ↓)

| Model | License | MASE ↓ | CRPS ↓ | Core-v3 vs. Model |
|---|---|---:|---:|---:|
| [Zitra-Titan-v3](https://huggingface.co/tomcupr/Zitra-Titan-v3) | Apache-2.0 | 0.6523 | 0.4420 | +0.4% / +1.2% |
| **⚡ Zitra-Core-v3** | **Apache-2.0** | **0.6551** | **0.4473** | — |
| Timestral-209M | Non-Comm. | 0.6575 | 0.4460 | **−0.4%** / +0.3% |
| Timestral-123M | Non-Comm. | 0.6589 | 0.4476 | **−0.6% / −0.1%** |
| t0-gamma | Apache-2.0 | 0.6608 | 0.4520 | **−0.9% / −1.0%** |
| TimesFM-3 | Non-Comm. | 0.6668 | 0.4557 | **−1.8% / −1.8%** |
| iAmTime | Apache-2.0 | 0.6795 | 0.4647 | **−3.6% / −3.7%** |
| Toto-2.0-2.5B | Apache-2.0 | 0.6956 | 0.4759 | **−5.8% / −6.0%** |
| Chronos-2 | Apache-2.0 | 0.6978 | 0.4854 | **−6.1% / −7.9%** |

Outperforms every third-party Apache-2.0 model of any size (including 10× larger `Toto-2.0-2.5B`), beats `Timestral-123M` on both metrics, and beats `Timestral-209M` on MASE (−0.93% MASE / −0.06% CRPS ex-Jena).

<details>
<summary><b>Real-world enterprise throughput &amp; raw CSVs (Rohlik · Zero-Shot)</b></summary>

- **High-volume SKU &amp; hourly ops (10,859 forecasts):** Forecasts **10,000 daily SKUs in 7.9 s (`1,265 SKUs/sec` on B300)** while staying within **0.5–1.6% WAPE** of `Titan-v3` on SKU and hourly warehouse ops (and beating `Titan-v3` on 28-day SKU MASE `1.260` vs. `1.284` and SQL `1.079` vs. `1.092`).
- **Public Rohlik Kaggle (`fev-bench` WQL):** Beats `Timestral-209M` across all 4 tasks — Orders 1D (**−6.0%**), Orders 1W (**−5.8%**), Sales 1D (**−4.8%**), Sales 1W (**−13.9%**), edging `Zitra-Titan-v3` on both Orders tasks.
- **Full benchmark CSVs:** [`gift_eval_results.csv`](gift_eval_results.csv) · [`fev_bench_results.csv`](fev_bench_results.csv).

</details>

---

## Speed, Hardware & When to Choose Core vs. Titan

Measured side-by-side in the same process on **NVIDIA B300 GPU** and **8 CPU cores (Intel Xeon 6776P, `fp32`, no GPU)**:

| Benchmark | Zitra-Core-v3 | [Zitra-Titan-v3](https://huggingface.co/tomcupr/Zitra-Titan-v3) | Core-v3 Advantage |
|---|---:|---:|---:|
| GPU Latency *(B300, single series)* | **28 ms** | 112 ms | **4.0× faster** |
| GPU Latency *(5 known covariates)* | **56 ms** | 197 ms | **3.5× faster** |
| CPU Latency *(8 cores, `fp32`)* | **86 ms** | 925 ms | **10.8× faster** |
| GPU VRAM | **1.0 GB** | 3.5 GB | **3.4× smaller** |
| Batch Throughput *(B300, B=64)* | **872 SKUs/s** | 114 SKUs/s | **7.7× higher** |
| Full `fev-bench` Runtime *(100 tasks)* | **1,807 s** | 5,719 s | **3.2× faster** |

`fast=True` trims single-series latency further to **26 ms** GPU and **65 ms** CPU — faster on 8 CPU cores than `Zitra-Titan-v3` on GPU. On large 10K-SKU daily batches, `Zitra-Core-v3` sustains **1,265 SKUs/s**.

- **Choose `Zitra-Core-v3`** for CPU-only, serverless, edge deployments, high-volume 100K–1M SKU pipelines, and latency-critical serving.
- **Choose [`Zitra-Titan-v3`](https://huggingface.co/tomcupr/Zitra-Titan-v3)** for maximum accuracy on high-stakes capacity, energy, and public-holiday planning.

---

## How It Works

- **Distilled from `Zitra-Titan-v3`:** Compresses the strengths of our 4-architecture flagship into a single compact `256M` Patch-Decoder backbone trained on strictly zero-overlap data, preserving sharp covariates and calibrated 21-quantile uncertainty (`0.01`–`0.99`).
- **Precision-safe & fast by default:** Runs short or high-offset series in `fp32` to eliminate low-precision drift, automatically switches to fast `bf16` attention on long histories (up to **16,384 steps**), and enables cross-series learning only when it improves internal backtests (`cross_learning="auto"`).
- **Zero deployment friction:** Ships as one self-contained `safetensors` file with exact singleton-vs-batch parity (`cross_learning=False`) and automatic handling of non-negative and integer series.

---

## License

**Apache License, Version 2.0** — 100% free for commercial and research use. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for full attributions.
