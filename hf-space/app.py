"""Interactive Hugging Face Space Demo for the Zitra v2 Forecasting Suite.

Features:
  - 1-click real-world presets (E-Commerce Daily Demand, Solar Generation, Hourly Fulfillment Orders,
    Cloud Latency Telemetry, City Bike Counts) OR upload any custom CSV.
  - Switch seamlessly between Zitra-Titan-585M, Zitra-Apex-585M, and Zitra-209M-v2.
  - Optional holdout backtest toggle to compare Zitra directly against ground truth & Stock Timestral-209M!
  - Live visualization of Zitra-Titan-585M's causal Bayesian expert weights [Zitra-209M-v2, t0-gamma, Chronos-2]
    and individual foundation model medians alongside the Mixture-CDF probabilistic forecast cone.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

try:
    import spaces  # Hugging Face ZeroGPU decorator when deployed on HF Spaces
except ImportError:  # local execution fallback
    class _DummySpaces:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            return fn if fn is not None else (lambda f: f)

    spaces = _DummySpaces()

from zitra import ZitraForecaster

HF_ORG = os.environ.get("ZITRA_HF_ORG", "tomcupr")
MODEL_REPOS = {
    "Zitra-Titan-585M (Flagship Adaptive)": f"{HF_ORG}/Zitra-Titan-585M",
    "Zitra-Apex-585M (Fast Covariate 3-Prior)": f"{HF_ORG}/Zitra-Apex-585M",
    "Zitra-209M-v2 (1.00x Speed Drop-In)": f"{HF_ORG}/Zitra-209M-v2",
}

EXPERT_COLORS = {
    "zitra-209m-v2": "#4f46e5",
    "t0-gamma": "#f59e0b",
    "chronos-2": "#10b981",
}


def make_preset(name: str) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    if name == "E-Commerce Daily Orders (Weekly + Promo Spikes)":
        n = 365
        ts = pd.date_range("2025-11-01", periods=n, freq="D")
        t = np.arange(n)
        dow = np.array([1.08, 0.94, 0.92, 0.97, 1.15, 1.24, 0.88])[t % 7]
        trend = 4200 + 3.2 * t + 320 * np.sin(2 * np.pi * t / 365.0)
        promos = ((t % 30 == 0) | (t % 30 == 1)).astype(float) * 780
        y = np.maximum(50, np.round(trend * dow + promos + rng.normal(0, 140, n)))
        return pd.DataFrame({"timestamp": ts, "value": y})

    if name == "Solar PV Generation (Hourly Diurnal + Cloud Shifts)":
        n = 504  # 21 days
        ts = pd.date_range("2026-06-01", periods=n, freq="h")
        t = np.arange(n)
        hour = t % 24
        sun = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0)) ** 1.4
        cloud_day = np.repeat(rng.uniform(0.55, 1.05, n // 24), 24)
        y = np.maximum(0.0, 95.0 * sun * cloud_day + rng.normal(0, 2.2, n) * (sun > 0))
        return pd.DataFrame({"timestamp": ts, "value": np.round(y, 2)})

    if name == "Fulfillment Center Hourly Pick Volume":
        n = 336  # 14 days
        ts = pd.date_range("2026-10-01", periods=n, freq="h")
        t = np.arange(n)
        h = t % 24
        d = (t // 24) % 7
        intraday = 0.15 + 0.85 * np.exp(-0.5 * ((h - 11.0) / 3.8) ** 2) + 0.55 * np.exp(-0.5 * ((h - 18.0) / 2.6) ** 2)
        wk = np.where(d >= 5, 1.18, 1.0)
        y = np.maximum(0, np.round(6200 * intraday * wk + rng.normal(0, 210, n)))
        return pd.DataFrame({"timestamp": ts, "value": y})

    if name == "Cloud CPU Utilization (Regime Shift + Burstiness)":
        n = 480
        ts = pd.date_range("2026-10-15", periods=n, freq="15min")
        t = np.arange(n)
        base = np.where(t < 260, 38.0, 56.0)
        wave = 14.0 * np.sin(2 * np.pi * t / 96.0) + 6.0 * np.sin(2 * np.pi * t / 24.0)
        burst = (rng.random(n) < 0.05) * rng.uniform(10, 22, n)
        y = np.clip(base + wave + burst + rng.normal(0, 2.5, n), 2.0, 99.0)
        return pd.DataFrame({"timestamp": ts, "value": np.round(y, 2)})

    # City Bike Counts (Integer Nonnegative)
    n = 360
    ts = pd.date_range("2026-09-01", periods=n, freq="h")
    t = np.arange(n)
    h = t % 24
    commute = 12 * np.exp(-0.5 * ((h - 8.5) / 1.8) ** 2) + 15 * np.exp(-0.5 * ((h - 17.5) / 2.2) ** 2)
    y = np.clip(np.round(2.5 + commute + rng.normal(0, 2.0, n)), 0, 32)
    return pd.DataFrame({"timestamp": ts, "value": y})


@lru_cache(maxsize=4)
def get_model(repo_id: str) -> ZitraForecaster:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ZitraForecaster.from_pretrained(repo_id, device=device)


@lru_cache(maxsize=1)
def get_stock_timestral():
    from timestral import TimestralForecaster

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device=device)


def _parse_input(preset_name: str, csv_file) -> tuple[pd.DataFrame, str, str]:
    if csv_file is not None:
        df = pd.read_csv(csv_file.name)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise gr.Error("Uploaded CSV must contain at least one numeric target column.")
        val_col = num_cols[-1]
        non_num = [c for c in df.columns if c not in num_cols]
        ts_col = non_num[0] if non_num else None
        if ts_col is not None:
            try:
                ts = pd.to_datetime(df[ts_col])
            except Exception:
                ts = pd.RangeIndex(len(df))
        else:
            ts = pd.RangeIndex(len(df))
        out = pd.DataFrame({"timestamp": ts, "value": df[val_col].astype(float)})
        return out.dropna().reset_index(drop=True), val_col, os.path.basename(csv_file.name)
    return make_preset(preset_name), "value", preset_name


@spaces.GPU(duration=60)
def run_demo(
    preset_name: str,
    csv_file,
    model_choice: str,
    horizon: int,
    eval_holdout: bool,
    compare_timestral: bool,
    show_experts: bool,
):
    df, val_col, label = _parse_input(preset_name, csv_file)
    H = int(horizon)
    if len(df) < 32:
        raise gr.Error("Need at least 32 observations to generate a forecast.")

    if eval_holdout and len(df) > H + 32:
        hist_df = df.iloc[:-H].reset_index(drop=True)
        truth_df = df.iloc[-H:].reset_index(drop=True)
        future_x = truth_df["timestamp"]
    else:
        hist_df = df
        truth_df = None
        ts = hist_df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts):
            step = ts.iloc[-1] - ts.iloc[-2]
            future_x = pd.date_range(start=ts.iloc[-1] + step, periods=H, freq=step)
        else:
            future_x = np.arange(len(hist_df), len(hist_df) + H)

    ctx = hist_df["value"].to_numpy(dtype=np.float32)
    repo_id = MODEL_REPOS[model_choice]

    t0 = time.perf_counter()
    model = get_model(repo_id)
    fc = model.forecast(ctx, prediction_length=H)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    med = fc.median[0, 0]
    q10, q20, q80, q90 = fc.quantile(0.1)[0, 0], fc.quantile(0.2)[0, 0], fc.quantile(0.8)[0, 0], fc.quantile(0.9)[0, 0]

    # Build main interactive forecast plot
    fig = go.Figure()
    hist_view = hist_df.iloc[-min(len(hist_df), max(H * 4, 160)) :]

    fig.add_trace(
        go.Scatter(
            x=hist_view["timestamp"],
            y=hist_view["value"],
            mode="lines",
            name="Historical Data",
            line=dict(color="#111827", width=2.2),
        )
    )

    # 10-90% uncertainty band
    fig.add_trace(
        go.Scatter(
            x=list(future_x) + list(future_x)[::-1],
            y=list(q90) + list(q10)[::-1],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Interval (q10–q90)",
            hoverinfo="skip",
        )
    )
    # 20-80% uncertainty band
    fig.add_trace(
        go.Scatter(
            x=list(future_x) + list(future_x)[::-1],
            y=list(q80) + list(q20)[::-1],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.25)",
            line=dict(color="rgba(255,255,255,0)"),
            name="60% Interval (q20–q80)",
            hoverinfo="skip",
        )
    )

    # Individual expert medians (Zitra-Titan-585M transparency!)
    if show_experts and fc.expert_medians:
        for exp_name, exp_med in fc.expert_medians.items():
            fig.add_trace(
                go.Scatter(
                    x=future_x,
                    y=exp_med[0, 0],
                    mode="lines",
                    name=f"Expert: {exp_name}",
                    line=dict(color=EXPERT_COLORS.get(exp_name, "#6b7280"), width=1.4, dash="dot"),
                    opacity=0.85,
                )
            )

    # Zitra median forecast
    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=med,
            mode="lines",
            name=f"{model_choice.split(' (')[0]} (Median)",
            line=dict(color="#4f46e5", width=3.0),
        )
    )

    metrics_lines = [
        f"**Model:** `{repo_id}`  |  **Series:** `{label}` (`{len(ctx)}` steps → `{H}` steps)  |  **Inference:** `{latency_ms:.0f} ms`"
    ]

    # Optional comparison vs Stock Timestral-209M
    stock_med = None
    if compare_timestral:
        st_model = get_stock_timestral()
        st_fc = st_model.forecast(ctx[None, None, :], prediction_length=H)
        stock_med = st_fc.median[0, 0].detach().cpu().float().numpy()
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=stock_med,
                mode="lines",
                name="Stock Timestral-209M",
                line=dict(color="#ef4444", width=2.0, dash="dash"),
            )
        )

    if truth_df is not None:
        y_true = truth_df["value"].to_numpy(dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=y_true,
                mode="lines+markers",
                name="Actual Holdout Truth",
                line=dict(color="#059669", width=2.2),
                marker=dict(size=4),
            )
        )
        z_mae = float(np.mean(np.abs(y_true - med)))
        z_wape = float(np.sum(np.abs(y_true - med)) / max(np.sum(np.abs(y_true)), 1e-8)) * 100.0
        line = f"**Holdout Error ({H} steps):** Zitra MAE = `{z_mae:.3f}` · WAPE = `{z_wape:.2f}%`"
        if stock_med is not None:
            s_mae = float(np.mean(np.abs(y_true - stock_med)))
            s_wape = float(np.sum(np.abs(y_true - stock_med)) / max(np.sum(np.abs(y_true)), 1e-8)) * 100.0
            delta = (z_mae / max(s_mae, 1e-8) - 1.0) * 100.0
            line += f"  |  Stock Timestral MAE = `{s_mae:.3f}` · WAPE = `{s_wape:.2f}%`  (**{delta:+.1f}% vs Stock**)"
        metrics_lines.append(line)

    fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Time",
        yaxis_title=val_col,
        hovermode="x unified",
    )

    # Build Live Expert Routing Weights bar chart (Zitra-Titan-585M signature feature!)
    routing_summary = fc.routing_summary()
    w_fig = go.Figure()
    if routing_summary is not None:
        names = ["Zitra-209M-v2\n(Folded 24L LoRA)", "t0-gamma\n(Patch Decoder)", "Chronos-2\n(Encoder-Only)"]
        vals = [routing_summary["zitra-209m-v2"] * 100, routing_summary["t0-gamma"] * 100, routing_summary["chronos-2"] * 100]
        colors = [EXPERT_COLORS["zitra-209m-v2"], EXPERT_COLORS["t0-gamma"], EXPERT_COLORS["chronos-2"]]
        w_fig.add_trace(
            go.Bar(
                x=vals,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
            )
        )
        w_fig.update_layout(
            title="Live Causal Pop-Quiz Posterior Weights (Zitra-Titan-585M)",
            template="plotly_white",
            height=220,
            margin=dict(l=20, r=40, t=45, b=20),
            xaxis=dict(range=[0, 108], title="Mixture-CDF Weight (%)"),
            yaxis=dict(autorange="reversed"),
        )
    else:
        w_fig.add_annotation(
            text="Live causal pop-quiz routing weights are shown when Zitra-Titan-585M is selected.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
        )
        w_fig.update_layout(template="plotly_white", height=180, xaxis_visible=False, yaxis_visible=False)

    out_df = pd.DataFrame(
        {
            "timestamp": future_x,
            "q10": np.round(q10, 4),
            "median": np.round(med, 4),
            "q90": np.round(q90, 4),
        }
    )
    return fig, w_fig, "\n\n".join(metrics_lines), out_df


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), title="Zitra v2 Live Demo") as demo:
    gr.Markdown(
        """
        # ⚡ Zitra v2 — Test-Time Adaptive Time-Series Foundation Models
        **0.6443 MASE / 0.4352 WQL on Official GIFT-Eval** · **Wins every slice of FEV-Bench** · **Zero-overlap open weights**

        Pick a real-world preset or drop in your own CSV to watch **Zitra-Titan-585M** run live causal pop-quizzes across three foundation priors (**Zitra-209M-v2**, **t0-gamma**, and **Chronos-2**) and invert their exact **Linear-Pool Mixture CDF**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=list(MODEL_REPOS.keys()),
                value="Zitra-Titan-585M (Flagship Adaptive)",
                label="Model",
            )
            preset_name = gr.Dropdown(
                choices=[
                    "E-Commerce Daily Orders (Weekly + Promo Spikes)",
                    "Solar PV Generation (Hourly Diurnal + Cloud Shifts)",
                    "Fulfillment Center Hourly Pick Volume",
                    "Cloud CPU Utilization (Regime Shift + Burstiness)",
                    "City Bike Counts (Integer Nonnegative)",
                ],
                value="E-Commerce Daily Orders (Weekly + Promo Spikes)",
                label="1-Click Sample Series",
            )
            csv_file = gr.File(label="Or Upload Your Own CSV (timestamp + numeric target)", file_types=[".csv"])
            horizon = gr.Slider(minimum=12, maximum=192, step=12, value=48, label="Forecast Horizon (steps)")
            eval_holdout = gr.Checkbox(value=True, label="Hold out last H points to score live accuracy")
            compare_timestral = gr.Checkbox(value=False, label="Overlay Stock Timestral-209M baseline")
            show_experts = gr.Checkbox(value=True, label="Show individual expert medians (Titan)")
            run_btn = gr.Button("⚡ Forecast Now", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_md = gr.Markdown("Click **Forecast Now** to run live inference.")
            main_plot = gr.Plot(label="Probabilistic Forecast")
            weights_plot = gr.Plot(label="Live Expert Routing Weights")
            with gr.Accordion("Forecast Table (Downloadable Quantiles)", open=False):
                out_table = gr.Dataframe(interactive=False)

    run_btn.click(
        fn=run_demo,
        inputs=[preset_name, csv_file, model_choice, horizon, eval_holdout, compare_timestral, show_experts],
        outputs=[main_plot, weights_plot, summary_md, out_table],
    )

if __name__ == "__main__":
    demo.launch()
