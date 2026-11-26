# ⚡ Zitra v3 — #1 Fully Open (Apache-2.0) Zero-Shot Forecasting Model

[![License](https://img.shields.io/badge/License-Apache%202.0-10b981)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Zitra--Titan--v3-4f46e5)](https://huggingface.co/tomcupr/Zitra-Titan-v3)
[![Live Demo](https://img.shields.io/badge/⚡%20Live%20Demo-HF%20Space-indigo)](https://huggingface.co/spaces/tomcupr/zitra-v3-demo)
[![GIFT-Eval](https://img.shields.io/badge/GIFT--Eval-0.6522%20MASE%20%2F%200.4422%20CRPS-4f46e5)](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
[![fev-bench](https://img.shields.io/badge/fev--bench-50.18%25%20SQL%20%2F%2080.2%25%20Win-10b981)](https://huggingface.co/spaces/autogluon/fev-bench)

**Zitra v3 (`Zitra-Titan-v3`)** is the **#1 fully open-source (Pure Apache-2.0) zero-shot time-series foundation model** across both **Official GIFT-Eval (97 configs)** and **`fev-bench` (100 tasks)** — with **zero benchmark overlap**, **built-in public holidays for 240+ countries**, and a single **`2.12 GB`** checkpoint (**`3.46 GB`** VRAM, fits any 6–8 GB consumer GPU).

![Zitra v3 Benchmarks on GIFT-Eval and fev-bench](assets/pareto_gift_fev.png)

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
print(fc.routing_summary())
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

## Verified Benchmarks (Single Frozen Checkpoint, Zero Leakage)

| Model | License | GIFT MASE ↓ | GIFT CRPS ↓ | fev SQL Win ↑ | fev SQL Skill ↑ | fev MASE Skill ↑ |
|---|---|---:|---:|---:|---:|---:|
| **[⚡ Zitra-Titan-v3](https://huggingface.co/tomcupr/Zitra-Titan-v3)** | **Apache-2.0** | **0.6522** | **0.4422** | **80.2%** (#6) | **50.18%** | **39.05%** |
| Timestral-209M | *Non-Commercial* | 0.6575 | 0.4460 | 79.4% (#7) | 50.39% | 39.49% |
| t0-gamma | Apache-2.0 | 0.6610 | 0.4520 | 73.8% (#12) | 49.42% | 37.99% |
| TimesFM-3 | *Non-Commercial* | 0.6670 | 0.4560 | 67.6% (#17) | 48.99% | 37.82% |
| Chronos-2 | Apache-2.0 | 0.6980 | 0.4850 | 57.9% (#23) | 47.28% | 35.50% |
| Toto-2.0-2.5B | Apache-2.0 | 0.6960 | 0.4760 | 56.9% (#25) | 44.42% | 32.54% |

- **GIFT-Eval (97 / 97):** #1 zero-shot open-weight model with any commercial license; beats `Timestral-209M` (−0.82% / −0.86%), `t0-gamma` (−1.33% / −2.17%), `TimesFM-3` (−2.22% / −3.03%), and `Chronos-2` (−6.56% / −8.82%).
- **`fev-bench` (100 / 100):** First and only standard Apache-2.0 model past 50% SQL skill (**54.52% WQL**, **79.4% MASE win rate**), ranking **#6 / 55 models** ahead of `Timestral-209M` (#7) on both SQL and MASE win rates.

### Real-World Enterprise Proof (Rohlik · 100% Zero-Shot)

![Zero-Shot Rohlik Retail and Fulfillment Benchmarks](assets/rohlik_benchmark.png)

- **Holiday weeks (12 European FCs, 2026 `D+7`):** Built-in `country=` holidays cut overall WAPE **4.62% → 3.88%** and slash 31 holiday weeks **13.58% → 7.81%** (**−24.9%** vs. Timestral-209M's 10.39%).
- **Production ops (8 tasks, 10,859 forecasts, no holidays):** Wins **6 / 8** tasks vs. Timestral-209M (**−0.92% WAPE**, up to **−5.10%** on 7-day hourly FC orders).
- **Public Rohlik Kaggle (`fev-bench`):** Cuts WQL vs. Timestral-209M by **−4.5% / −5.5%** (Orders 1D/1W) and **−7.1% / −15.4%** (SKU Sales 1D/1W).

---

## Architecture & Systems Engineering

| Expert | Architecture | Params | Role & Apache-2.0 Base |
|---|---|---:|---|
| **Zitra-t0-256M** | Patch-Decoder | 255.6M | Dyn32K soup · 16K ctx ([t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma)) |
| **Zitra-Chronos2** | T5 Encoder | 119.5M | Covariate soup ([chronos-2](https://huggingface.co/amazon/chronos-2)) |
| **Zitra-PatchTST** | 30L Conformer | 384.6M | Long univariate · skipped on covs ([PatchTST-FM-r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2)) |
| **iAmTime-302M** | T5 4-Expert MoE | 302.1M | Native `arcsinh` MoE ([iAmTime](https://huggingface.co/anishsaha/iAmTime-base-forecast)) |

- **Fused-Batch Pop-Quiz Execution (`2.6×` speedup: 321 ms → 122 ms):** Packs `[main, -main, bt0, bt1]` into fused padded GPU passes per expert with **0 learned router weights** (plus opt-in single-pass `fast=True` at 102–115 ms).
- **Single `2.12 GB` `bfloat16` File + Mixed-Precision Discipline (`3.46 GB` VRAM):** Hydrates 30-layer Conformer `PatchTST` to `fp32` under `bf16` autocast (`+0.007%` FP32 parity) and folds all LoRAs algebraically into base weights (zero adapter overhead).
- **16K Dynamic RoPE, Zero-Compute Covariate Gating & Mixture-CDF Inversion:** Dynamic RoPE `scale_base` (`512/1536/2048`) up to 16,384 steps, skips `PatchTST` on known covariates, self-gates cross-learning via causal backtests, and inverts `F_mix(y) = Σ w_e F_e(y)` across 21 quantiles with **≤ 2e-5** batch/singleton invariance.

---

## Live Demo & License

Try **Zitra v3** live on [Hugging Face Spaces (`tomcupr/zitra-v3-demo`)](https://huggingface.co/spaces/tomcupr/zitra-v3-demo), or run locally with `pip install "zitra-forecast[demo] @ git+https://github.com/tomascupr/zitra-forecast.git" && python hf-space/app.py`.

**Apache License, Version 2.0** — 100% free for commercial and research use. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
