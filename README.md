# ⚡ Zitra v3 — #1 Fully Open (Apache-2.0) Forecasting Models

[![License](https://img.shields.io/badge/License-Apache%202.0-10b981)](LICENSE)
[![Titan-v3](https://img.shields.io/badge/🤗%20Flagship-Zitra--Titan--v3-0f172a)](https://huggingface.co/tomcupr/Zitra-Titan-v3)
[![Core-v3](https://img.shields.io/badge/🤗%20Fast%20256M-Zitra--Core--v3-4f46e5)](https://huggingface.co/tomcupr/Zitra-Core-v3)
[![Live Demo](https://img.shields.io/badge/⚡%20Live%20Demo-HF%20Space-indigo)](https://huggingface.co/spaces/tomcupr/zitra-v3-demo)

**Zitra v3** is a two-tier family of **100% Apache-2.0** zero-shot probabilistic time-series foundation models — ranked **#1 and #2 among all fully open (Apache-2.0) models** across both **GIFT-Eval** and **`fev-bench`**, with built-in public holidays for 240+ countries:

| Model | Params | GPU / 8-Core CPU | VRAM | GIFT-Eval (MASE ↓ / CRPS ↓) | `fev-bench` (SQL ↑ / Win ↑) | Best For |
|---|---:|---:|---:|---|---|---|
| **[⚡ Zitra-Titan-v3](https://huggingface.co/tomcupr/Zitra-Titan-v3)** | 1.06B | 112 ms / 925 ms | 3.5 GB | **0.6523 / 0.4420** | **50.30%** / **80.1%** (#6) | Flagship accuracy · holiday & capacity planning |
| **[⚡ Zitra-Core-v3](https://huggingface.co/tomcupr/Zitra-Core-v3)** | **256M** | **28 ms / 86 ms** | **1.0 GB** | **0.6551 / 0.4473** | **49.69%** / **75.4%** (#11) | High-throughput SKUs (`1,265/s`) · CPU & edge |

![Zitra-Core-v3 and Zitra-Titan-v3 Benchmarks](assets/zitra_core_v3_benchmarks.png)

---

## Quickstart

```bash
pip install "zitra-forecast @ git+https://github.com/tomascupr/zitra-forecast.git"
```

```python
import numpy as np
from zitra import ZitraForecaster

# Flagship accuracy: "tomcupr/Zitra-Titan-v3"  |  4x faster GPU / 11x CPU: "tomcupr/Zitra-Core-v3"
model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Core-v3")
ctx = np.random.default_rng(42).standard_normal(512).cumsum()
fc = model.forecast(ctx, prediction_length=96)

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

## Verified Benchmarks (Single Frozen Checkpoints, Zero Leakage)

| Model | License | GIFT MASE ↓ | GIFT CRPS ↓ | fev SQL Win ↑ | fev SQL Skill ↑ | fev Runtime ↓ |
|---|---|---:|---:|---:|---:|---:|
| **[⚡ Zitra-Titan-v3](https://huggingface.co/tomcupr/Zitra-Titan-v3)** *(1.06B)* | **Apache-2.0** | **0.6523** | **0.4420** | **80.1%** (#6) | **50.30%** | 5,719 s |
| Timestral-209M | Non-Comm. | 0.6575 | 0.4460 | 78.7% (#7) | 50.39% | 3,827 s |
| **[⚡ Zitra-Core-v3](https://huggingface.co/tomcupr/Zitra-Core-v3)** *(256M)* | **Apache-2.0** | **0.6551** | **0.4473** | **75.4%** (#11) | **49.69%** | **1,807 s** |
| t0-gamma *(256M)* | Apache-2.0 | 0.6608 | 0.4520 | 72.9% (#13) | 49.42% | 15,862 s |
| TimesFM-3 | Non-Comm. | 0.6668 | 0.4557 | 66.8% (#18) | 48.99% | 8,673 s |
| Chronos-2 *(120M)* | Apache-2.0 | 0.6978 | 0.4854 | 56.9% (#24) | 47.28% | 2,738 s |
| Toto-2.0-2.5B *(2.5B)* | Apache-2.0 | 0.6956 | 0.4759 | 55.9% (#25) | 44.42% | 13,835 s |

- **`Zitra-Titan-v3`:** `#1` fully open (Apache-2.0) model on both GIFT-Eval (`−0.8% / −0.9%` vs. `Timestral-209M`) and `fev-bench` (`#5` WQL win rate `78.9%`, `#6` SQL `80.1%`, `#6` MASE `79.4%` across 56 models — beating `Timestral-209M` on all 78 zero-overlap tasks while running **2.8× faster than `t0-gamma`**, **2.4× faster than `Toto-2.0-2.5B`**, and **1.5× faster than `TimesFM-3`**).
- **`Zitra-Core-v3`:** `#1` compact (`256M`) Apache-2.0 model on both leaderboards — beating `t0-gamma` (`63/100` `fev-bench` wins), `TimesFM-3`, `Chronos-2`, and `Toto-2.0-2.5B` while running **8.8× faster than `t0-gamma`**, **4.8× faster than `TimesFM-3`**, and **2.1× faster than `Timestral-209M`**.

### Real-World Enterprise Proof (Rohlik · Zero-Shot)

![Zero-Shot Rohlik Retail and Fulfillment Benchmarks](assets/rohlik_benchmark.png)

- **Holiday weeks (12 European FCs, 2026 `D+7`):** `Zitra-Titan-v3` + built-in `country=` holidays cut overall WAPE **4.62% → 3.84%** and slash 31 holiday weeks **13.58% → 7.78%** (**−25.2%** vs. `Timestral-209M`'s 10.39%).
- **High-volume SKU & hourly ops (10,859 forecasts):** `Zitra-Titan-v3` beats `Timestral-209M` on **6 / 8** tasks (`−4.2%` / `−3.7%` on hourly FC orders); `Zitra-Core-v3` forecasts **10,000 SKUs in 7.9 s (`1,265 SKUs/sec`)** within **0.5–1.6% WAPE** of Titan (and beats Titan on 28-day SKU MASE & SQL).
- **Public Rohlik Kaggle (`fev-bench`):** Both models beat `Timestral-209M` on all 4 tasks (`Titan-v3` WQL: **−3.9% / −5.5%** Orders 1D/1W, **−7.2% / −15.4%** Sales 1D/1W; `Core-v3`: **−6.0% / −5.8%** Orders, **−4.8% / −13.9%** Sales).

---

## Live Demo & License

Try both **`Zitra-Titan-v3`** and **`Zitra-Core-v3`** live on [Hugging Face Spaces (`tomcupr/zitra-v3-demo`)](https://huggingface.co/spaces/tomcupr/zitra-v3-demo), or run locally with `pip install "zitra-forecast[demo] @ git+https://github.com/tomascupr/zitra-forecast.git" && python hf-space/app.py`.

**Apache License, Version 2.0** — 100% free for commercial and research use. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
