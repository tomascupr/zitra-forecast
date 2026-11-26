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

# ⚡ Zitra-Titan-v3

**#1 Fully Open (Apache-2.0) Zero-Shot Time-Series Foundation Model** across both **Official GIFT-Eval (97 configs)** and **`fev-bench` (100 tasks)** — with **zero benchmark overlap**.

**2.12 GB single file** · **3.46 GB GPU VRAM** *(any 6–8 GB GPU)* · **122 ms latency** *(B300)* · **Built-in public holidays for 240+ countries**

[**Live Demo Space**](https://huggingface.co/spaces/tomcupr/zitra-v3-demo) · [**GitHub SDK (`zitra-forecast`)**](https://github.com/tomascupr/zitra-forecast) · [**GIFT-Eval CSV**](gift_eval_results.csv) · [**fev-bench CSV**](fev_bench_results.csv)

![Zitra v3 Benchmarks on GIFT-Eval and fev-bench](https://raw.githubusercontent.com/tomascupr/zitra-forecast/main/assets/pareto_gift_fev.png)

---

## Quickstart

```bash
pip install "zitra-forecast @ git+https://github.com/tomascupr/zitra-forecast.git"
```

```python
import numpy as np
from zitra import ZitraForecaster

model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Titan-v3")
ctx = np.random.default_rng(42).standard_normal(512).cumsum()
fc = model.forecast(ctx, prediction_length=96)  # fast=True for ~102ms

median, q10, q90 = fc.median, fc.quantile(0.1), fc.quantile(0.9)
print(fc.routing_summary())  # Live causal pop-quiz expert weights
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

Every number below comes from this single frozen checkpoint (`frozen.json` SHA-256 `f61bde12f03830e9`) with strictly causal zero-leakage evaluation.

### Official GIFT-Eval (97 / 97 configs · lower is better ↓)

| Model | License | MASE ↓ | CRPS ↓ | Zitra Gain (MASE / CRPS) |
|---|---|---:|---:|---:|
| **⚡ Zitra-Titan-v3** | **Apache-2.0** | **0.6522** | **0.4422** | — |
| Timestral-209M | *Non-Commercial* | 0.6575 | 0.4460 | −0.82% / −0.86% |
| t0-gamma | Apache-2.0 | 0.6610 | 0.4520 | −1.33% / −2.17% |
| TimesFM-3 | *Non-Commercial* | 0.6670 | 0.4560 | −2.22% / −3.03% |
| Toto-2.0-2.5B | Apache-2.0 | 0.6960 | 0.4760 | −6.29% / −7.10% |
| Chronos-2 | Apache-2.0 | 0.6980 | 0.4850 | −6.56% / −8.82% |

*#1 among all zero-shot open-weight models with any commercial license on GIFT-Eval (all higher entries are closed agents/APIs, non-commercial, or fine-tuned on GIFT-Eval).*

### Official `fev-bench` (100 / 100 tasks · higher is better ↑)

| Rank | Model | License | SQL Win Rate ↑ | SQL Skill ↑ | MASE Skill ↑ |
|---:|---|---|---:|---:|---:|
| **#6** | **⚡ Zitra-Titan-v3** | **Apache-2.0** | **80.2%** | **50.18%** | **39.05%** |
| #7 | Timestral-209M | *Non-Commercial* | 79.4% | 50.39% | 39.49% |
| #12 | t0-gamma | Apache-2.0 | 73.8% | 49.42% | 37.99% |
| #17 | TimesFM-3 | *Non-Commercial* | 67.6% | 48.99% | 37.82% |
| #23 | Chronos-2 | Apache-2.0 | 57.9% | 47.28% | 35.50% |
| #25 | Toto-2.0-2.5B | Apache-2.0 | 56.9% | 44.42% | 32.54% |

*First and only standard Apache-2.0 model past 50% SQL skill. Also achieves **54.52% WQL skill** and **79.4% MASE win rate** (#6 / 55 models, ahead of Timestral-209M's 78.7% at #7). Ranks #1–#5 are closed APIs, non-commercial, or custom non-Apache licenses.*

<details>
<summary><b>Per-dataset highlights &amp; raw CSVs</b></summary>

- **GIFT-Eval vs. Timestral-209M (`−1.61% MASE / −1.34% CRPS` ex-Jena):** `solar` (−26.3% / −24.3%), `temperature_rain` (−10.5% / −17.1%), `m4_hourly` (−9.0% CRPS), `bizitobs_application` (−2.7% / −7.5%), `kdd_cup_2018` (−4.5% / −5.3%), `loop_seattle` (−3.4% / −4.1%), `hospital` (−0.9% / −3.0%). Full 97-row CSV: [`gift_eval_results.csv`](gift_eval_results.csv).
- **`fev-bench` vs. Timestral-209M (51 / 100 SQL wins):** `past_cov_only` (−3.84%), `epf_de` (−8.37%), `favorita_stores_1W` (−12.28%), `rohlik_sales_1W` (−11.89%), `uk_covid_nation_1W` (−22.61%), `solar_1W` (−12.92%). Full 100-row CSV: [`fev_bench_results.csv`](fev_bench_results.csv).

</details>

### Real-World Enterprise Proof (Rohlik · 100% Zero-Shot)

![Zero-Shot Rohlik Retail and Fulfillment Benchmarks](https://raw.githubusercontent.com/tomascupr/zitra-forecast/main/assets/rohlik_benchmark.png)

- **Holiday weeks (12 European FCs, 2026 `D+7`):** Built-in `country=` holidays cut overall WAPE **4.62% → 3.88%** and slash 31 holiday weeks **13.58% → 7.81%** (**−24.9%** vs. Timestral-209M's 10.39%).
- **Production ops (8 tasks, 10,859 forecasts, no holidays):** Wins **6 / 8** tasks vs. Timestral-209M (**−0.92% WAPE**, up to **−5.10%** on 7-day hourly FC orders).
- **Public Rohlik Kaggle (`fev-bench`):** Cuts WQL vs. Timestral-209M by **−4.5% / −5.5%** (Orders 1D/1W) and **−7.1% / −15.4%** (SKU Sales 1D/1W).

---

## Architecture & Systems Engineering

Zitra-Titan-v3 unifies **four genuinely orthogonal Apache-2.0 foundation architectures** inside one self-contained `2.12 GB` checkpoint and adapts at inference time with **zero learned router weights**:

| Expert | Architecture | Params | Role & Apache-2.0 Base |
|---|---|---:|---|
| **Zitra-t0-256M** | Patch-Decoder | 255.6M | Dyn32K soup · 16K ctx ([t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma)) |
| **Zitra-Chronos2** | T5 Encoder | 119.5M | Covariate soup ([chronos-2](https://huggingface.co/amazon/chronos-2)) |
| **Zitra-PatchTST** | 30L Conformer | 384.6M | Long univariate · skipped on covs ([PatchTST-FM-r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2)) |
| **iAmTime-302M** | T5 4-Expert MoE | 302.1M | Native `arcsinh` MoE ([iAmTime](https://huggingface.co/anishsaha/iAmTime-base-forecast)) |

**Engineering highlights:**

- **Fused-Batch Pop-Quiz Execution (`2.6×` speedup: 321 ms → 122 ms):** Packs symmetric sign-averaging (`+x`, `-x`) and two causal rolling backtests (`bt0`, `bt1`) into fused padded GPU passes per expert — zero learned router weights, zero benchmark overfitting, plus opt-in single-pass `fast=True` (102–115 ms).
- **Single `2.12 GB` `bfloat16` File + Mixed-Precision Conformer Hydration:** Compresses all 1.06B params into one `safetensors` file (3.46 GB VRAM). At load time, 30-layer Conformer `PatchTST` hydrates to `fp32` under `bf16` autocast for exact **+0.007%** FP32 parity without `bf16` weight drift.
- **Zero Runtime LoRA Overhead + 16K Dynamic RoPE:** All post-trained LoRAs are algebraically folded (`W' = W + (α/r)BA`) and souped across seeds. Dynamic RoPE `scale_base` (512 / 1536 / 2048) extends covariate context to **16,384 steps** with zero short-horizon drift.
- **Zero-Compute Covariate Gating & Structural Guards:** Skips `PatchTST` entirely when known covariates are present, enables cross-series attention only when it beats independent causal backtests (`cross_learning="auto"`), and guards monotonic/cumulative `arcsinh` tails.
- **Exact Mixture-CDF Inversion & Strict Batch Invariance:** Inverts `F_mix(y) = Σ w_e F_e(y)` directly across 21 quantiles for outlier-proof uncertainty, with attention-mask isolation guaranteeing **≤ 2e-5** max diff solo vs. batched.

### Hardware & Runtime Profile

| Specification | Value | Details |
|---|---|---|
| **Checkpoint** | **2.12 GB** | Single `bfloat16` `safetensors` file (vendored `_iamtime/` included) |
| **Parameters** | **1.06B** | 1,061,831,429 total · 0 learned router parameters |
| **GPU VRAM** | **3.46 GB** | Standard inference *(fits any 6–8 GB GPU; 6–12 GB for 16K batches)* |
| **Pop-Quiz** | **122 ms** | B300 univariate median · 174 ms known covariates (H=48) |
| **Fast mode** | **102 ms** | B300 single-pass (`fast=True`: 102 ms cov · 115 ms univ) |
| **Max context** | **16,384** | Dynamic RoPE (8,192 default) · 21 quantiles (0.01–0.99 grid) |

---

## License

**Apache License, Version 2.0** — 100% free for commercial and research use. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for full attributions.
