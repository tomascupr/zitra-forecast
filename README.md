# ⚡ Zitra v2 — State-of-the-Art Probabilistic Time-Series Foundation Models

[![Hugging Face](https://img.shields.io/badge/🤗%20Models-Zitra%20v2-yellow)](https://huggingface.co/tomcupr)
[![Live Demo](https://img.shields.io/badge/⚡%20Live%20Demo-HF%20Space-indigo)](https://huggingface.co/spaces/tomcupr/zitra-v2-demo)
[![GIFT-Eval](https://img.shields.io/badge/GIFT--Eval-0.6443%20MASE%20%2F%200.4352%20CRPS-4f46e5)](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
[![fev-bench](https://img.shields.io/badge/fev--bench-50.61%20SQL%20%2F%2039.85%20MASE-10b981)](https://huggingface.co/spaces/autogluon/fev-bench)

**Zitra v2** is a three-tier family of open-weight time-series forecasting models setting a new state of the art across **Official GIFT-Eval (97 configs)**, **fev-bench (100 tasks)**, **enterprise retail & fulfillment operations (Rohlik)**, and **cryptographically sealed live prospective forecasts** — with **zero benchmark overlap (`trained_on_benchmark_datasets = []`)**.

![Zitra v2 Benchmarks on GIFT-Eval and fev-bench](assets/pareto_gift_fev.png)

---

## Choose Your Zitra v2 Model

| Model | Params | B300 Latency | GIFT-Eval (MASE / CRPS ↓) | fev-bench Skill (MASE / SQL / WQL ↑) | Best For |
|---|---:|---:|---:|---:|---|
| **[Zitra-Titan-585M](https://huggingface.co/tomcupr/Zitra-Titan-585M)** | 584.4M | 251 ms (`9.7×`) | **`0.6443 / 0.4352`** | `39.71` / **`50.61 / 54.95`** | **Flagship — highest accuracy & calibration everywhere** |
| **[Zitra-Apex-585M](https://huggingface.co/tomcupr/Zitra-Apex-585M)** | 584.5M | 84 ms (`3.2×`) | `0.6569 / 0.4469` | **`39.85`** / `50.55 / 54.94` | **Fast 3-prior specialist for covariates, retail & hourly ops** |
| **[Zitra-209M-v2](https://huggingface.co/tomcupr/Zitra-209M-v2)** | 209.3M | **26 ms (`1.0×`)** | `0.6578 / 0.4451` | `39.56 / 50.45 / 54.81` | **Exact 209M zero-overhead drop-in upgrade to Timestral-209M** |
| *Timestral-209M (baseline)* | *209.3M* | *26 ms (`1.0×`)* | *0.6575 / 0.4460* | *39.49 / 50.39 / 54.63* | *Reference baseline* |

---

## 5-Line Quickstart

```bash
pip install "zitra-forecast[all] @ git+https://github.com/tomascupr/zitra-forecast.git"
```

```python
import numpy as np
from zitra import ZitraForecaster

model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Titan-585M")

context = np.random.default_rng(42).standard_normal(512).cumsum()
fc = model.forecast(context, prediction_length=96)

median = fc.median                      # (1, 1, 96)
q10, q90 = fc.quantile(0.1), fc.quantile(0.9)
print("Live pop-quiz expert weights:", fc.routing_summary())
```

### Pandas DataFrame API (with Optional Past & Known-Future Covariates)

```python
import pandas as pd
from zitra import ZitraForecaster

model = ZitraForecaster.from_pretrained("tomcupr/Zitra-Titan-585M")
pred_df = model.predict_df(
    df,
    prediction_length=28,
    id_column="item_id",
    timestamp_column="timestamp",
    target="target",
    quantile_levels=[0.1, 0.5, 0.9],
)
```

---

## Key Engineering Breakthroughs

1. **Algebraically Folded 24-Layer FullLoRA Soup (`Zitra-209M-v2`):** Five full-depth rank-16 LoRA runs (`wQKV`, `wO`, `mlp.w3`) averaged into a variance-reduced model soup and algebraically folded directly back into exact stock `TimestralModel` matrices (`max_abs_diff = 2.15e-6`) — **exact `1.003×` stock latency**.
2. **Causal Test-Time Pop-Quiz Routing (`Zitra-Titan-585M`):** Runs two covariate-preserving rolling historical backtests across three orthogonal foundation priors (`Zitra-209M-v2`, `t0-gamma` with full official `a2` flags, and `Chronos-2`), forming an unclamped Bayesian posterior shrunk by exact effective backtest sample size `n_eff / (n_eff + κ)` with **zero learned router weights**.
3. **Exact Linear-Pool Mixture-CDF Inversion:** Inverts `F_mix(y) = Σ w_e F_e(y)` at every decile — guaranteeing a bounded influence function against minority-expert hallucinations and automatic epistemic uncertainty widening when experts disagree.

## Real-World Retail & Fulfillment Proof (Rohlik, 100% Zero-Shot)

![Zero-Shot Rohlik Retail and Fulfillment Benchmarks](assets/rohlik_benchmark.png)

- **Public Rohlik Kaggle (fev-bench):** `Zitra-Titan-585M` reduces error by **`−3.99% MASE / −3.92% WQL`** on Rohlik Orders and **`−3.74% MASE / −7.21% WQL`** on Rohlik SKU Sales.
- **Internal 2026 Rohlik Production Holdout (8 operational tasks, 10,859 forecasts):** `Zitra-Titan-585M` wins **7/8 tasks** (`−0.83% MASE / −0.99% WQL / −1.05% WAPE`); `Zitra-Apex-585M` hits **`−1.28% WAPE`** overall and **`−3.48% MASE / −3.83% WQL`** on 24h hourly FC orders.

## Interactive Demo

Run the interactive Gradio demo locally or try it on [Hugging Face Spaces](https://huggingface.co/spaces/tomcupr/zitra-v2-demo):

```bash
pip install "zitra-forecast[demo] @ git+https://github.com/tomascupr/zitra-forecast.git"
python hf-space/app.py
```

## License

`Zitra-209M-v2` is derived from `mistralai/Timestral-209M` (Modified MIT). `Zitra-Apex-585M` and `Zitra-Titan-585M` bundle `Zitra-209M-v2` (Modified MIT), `t0-gamma` (Apache-2.0), and `Chronos-2` (Apache-2.0). Commercial use is subject to upstream component licenses.
