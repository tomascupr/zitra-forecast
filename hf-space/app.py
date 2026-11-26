"""Interactive Hugging Face Space Demo for Zitra v3 (100% Pure Apache-2.0).

Features:
  - 1-click real-world presets (Grocery Daily Orders with Public Holidays, Hourly Fulfillment Center,
    Solar PV Generation, Cloud CPU Telemetry, City Bike Counts) OR upload any custom CSV.
  - Built-in Opt-In Public Holiday & Calendar Covariates for 240+ countries & regional subdivisions
    (CZ, DE-BY, DE-BE, AT, HU, RO, US, GB, etc.) — watch how turning holidays ON automatically
    shifts Pop-Quiz routing 100% onto covariate-native Zitra-t0 + Zitra-Chronos2!
  - Optional holdout backtest toggle to score live MAE/WAPE against ground truth & Stock t0-gamma.
  - Live visualization of Zitra-Titan-v3's causal Bayesian expert weights
    [Zitra-t0-256M, Zitra-Chronos2-120M, Zitra-PatchTST-385M] and individual foundation medians
    alongside the Linear-Pool Mixture-CDF probabilistic forecast cone.
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

from zitra import ZitraForecaster, build_holiday_covariates

HF_ORG = os.environ.get("ZITRA_HF_ORG", "tomcupr")
DEFAULT_REPO = os.environ.get("ZITRA_V3_REPO", f"{HF_ORG}/Zitra-Titan-v3")

EXPERT_COLORS = {
    "t0": "#4f46e5",
    "chronos2": "#10b981",
    "patchtst": "#f59e0b",
    "iamtime": "#ec4899",
    "zitra-t0-256m": "#4f46e5",
    "zitra-chronos2-120m": "#10b981",
    "zitra-patchtst-385m": "#f59e0b",
}

EXPERT_LABELS = {
    "t0": "Zitra-t0-256M-Dyn32K\n(Patch Decoder · Apache-2.0)",
    "chronos2": "Zitra-Chronos2-120M\n(T5 Encoder · Apache-2.0)",
    "patchtst": "Zitra-PatchTST-385M\n(30L Conformer · Apache-2.0)",
    "iamtime": "iAmTime-302M\n(T5 4-MoE · Apache-2.0)",
    "zitra-t0-256m": "Zitra-t0-256M\n(Patch Decoder · Apache-2.0)",
    "zitra-chronos2-120m": "Zitra-Chronos2-120M\n(T5 Encoder · Apache-2.0)",
    "zitra-patchtst-385m": "Zitra-PatchTST-385M\n(30L Conformer · Apache-2.0)",
}

COUNTRY_CHOICES = [
    ("Off (No holiday covariates)", "NONE"),
    ("Czechia (CZ)", "CZ"),
    ("Germany — Bavaria / Munich (DE-BY)", "DE-BY"),
    ("Germany — Berlin (DE-BE)", "DE-BE"),
    ("Germany — Hesse / Frankfurt (DE-HE)", "DE-HE"),
    ("Austria (AT)", "AT"),
    ("Hungary (HU)", "HU"),
    ("Romania (RO)", "RO"),
    ("United States (US)", "US"),
    ("United Kingdom (GB)", "GB"),
]


def make_preset(name: str) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    if name == "Grocery Daily Orders (Czech Public Holidays + Weekly Cycle)":
        n = 420
        ts = pd.date_range("2025-10-01", periods=n, freq="D")
        t = np.arange(n)
        dow = np.array([1.10, 0.94, 0.92, 0.98, 1.18, 1.26, 0.86])[t % 7]
        trend = 14500 + 8.5 * t + 950 * np.sin(2 * np.pi * t / 365.0)
        hol = build_holiday_covariates(ts, "CZ")
        # Realistic grocery holiday pattern: huge pre-holiday surge, sharp holiday dip, post-holiday rebound
        hol_mult = 1.0 - 0.38 * hol[0] + 0.24 * hol[1] + 0.08 * hol[2]
        y = np.maximum(500, np.round(trend * dow * hol_mult + rng.normal(0, 320, n)))
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


@lru_cache(maxsize=2)
def get_model(repo_id: str) -> ZitraForecaster:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ZitraForecaster.from_pretrained(repo_id, device=device)


@lru_cache(maxsize=1)
def get_stock_t0():
    from t0 import T0Forecaster

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").to(device).eval()


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
    country_code: str,
    horizon: int,
    eval_holdout: bool,
    compare_stock_t0: bool,
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

    known_cov = None
    holiday_note = "Off"
    if country_code and country_code != "NONE":
        ts_hist = hist_df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts_hist):
            full_ts = pd.DatetimeIndex(ts_hist).append(pd.DatetimeIndex(future_x))
            known_cov = build_holiday_covariates(full_ts, country_code)[None, :, :]  # (1, 5, T+H)
            n_fut_hol = int(known_cov[0, 0, len(ctx) :].sum())
            holiday_note = f"`{country_code}` (5 calendar/holiday channels · {n_fut_hol} holiday steps in horizon)"
        else:
            holiday_note = "Ignored (series has no datetime timestamps)"

    t0_s = time.perf_counter()
    model = get_model(DEFAULT_REPO)
    fc = model.forecast(ctx, prediction_length=H, known_covariates=known_cov)
    latency_ms = (time.perf_counter() - t0_s) * 1000.0

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

    # Shade public holidays in horizon when enabled
    if known_cov is not None and pd.api.types.is_datetime64_any_dtype(hist_df["timestamp"]):
        fut_hol_flags = known_cov[0, 0, len(ctx) :]
        for idx_h, flag in enumerate(fut_hol_flags):
            if flag > 0.5:
                t_val = pd.Timestamp(future_x.iloc[idx_h] if hasattr(future_x, "iloc") else future_x[idx_h])
                fig.add_vrect(
                    x0=t_val - pd.Timedelta(hours=10),
                    x1=t_val + pd.Timedelta(hours=10),
                    fillcolor="rgba(239, 68, 68, 0.12)",
                    line_width=0,
                    annotation_text="Holiday",
                    annotation_position="top left",
                    annotation_font_size=9,
                )

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

    if show_experts and fc.expert_medians:
        routing_summary = fc.routing_summary() or {}
        for exp_name, exp_med in fc.expert_medians.items():
            if routing_summary.get(exp_name, 1.0) < 1e-4:
                continue  # skip gated-off univariate expert on known-holiday/short regimes
            fig.add_trace(
                go.Scatter(
                    x=future_x,
                    y=exp_med[0, 0],
                    mode="lines",
                    name=f"Expert: {exp_name}",
                    line=dict(color=EXPERT_COLORS.get(exp_name, "#6b7280"), width=1.5, dash="dot"),
                    opacity=0.85,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=med,
            mode="lines",
            name="Zitra-Titan-v3 (Median)",
            line=dict(color="#4f46e5", width=3.0),
        )
    )

    metrics_lines = [
        f"**Model:** `{DEFAULT_REPO}` *(100% Apache-2.0)*  |  **Series:** `{label}` (`{len(ctx)}` → `{H}`)  |  **Holidays:** {holiday_note}  |  **Inference:** `{latency_ms:.0f} ms`"
    ]

    stock_med = None
    if compare_stock_t0:
        from t0 import TimeSeries, batch_series

        st_model = get_stock_t0()
        device = next(st_model.parameters()).device
        with torch.inference_mode():
            if known_cov is not None:
                ts_in = TimeSeries.from_array(
                    torch.from_numpy(ctx[None, None, :]),
                    future_covariates=torch.from_numpy(known_cov),
                    horizon=H,
                )
                batch = TimeSeries.batch([ts_in]).to(device)
                st_out = st_model.predict(
                    batch, horizon=H, context_length=batch.seq_len - H,
                    quantile_levels=[0.5], trim_leading_missing=True, symmetric_averaging=True,
                    clip_nonnegative=True, snap_integers=True,
                ).quantiles.float()
            else:
                arr, mask, ids = batch_series([torch.from_numpy(ctx[None, :])])
                st_out = st_model.predict(
                    arr.to(device), horizon=H, mask=mask.to(device), group_ids=ids.to(device),
                    quantile_levels=[0.5], trim_leading_missing=True, symmetric_averaging=True,
                    clip_nonnegative=True, snap_integers=True,
                ).quantiles.float()
        stock_med = st_out.reshape(H).detach().cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=future_x,
                y=stock_med,
                mode="lines",
                name="Stock t0-gamma (Apache-2.0)",
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
        line = f"**Holdout Error ({H} steps):** Zitra-v3 MAE = `{z_mae:.3f}` · WAPE = `{z_wape:.2f}%`"
        if stock_med is not None:
            s_mae = float(np.mean(np.abs(y_true - stock_med)))
            s_wape = float(np.sum(np.abs(y_true - stock_med)) / max(np.sum(np.abs(y_true)), 1e-8)) * 100.0
            delta = (z_mae / max(s_mae, 1e-8) - 1.0) * 100.0
            line += f"  |  Stock t0-gamma MAE = `{s_mae:.3f}` · WAPE = `{s_wape:.2f}%`  (**{delta:+.1f}% vs Stock**)"
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

    routing_summary = fc.routing_summary()
    w_fig = go.Figure()
    if routing_summary is not None:
        keys = list(routing_summary.keys())
        names = [EXPERT_LABELS.get(k, k) for k in keys]
        vals = [routing_summary[k] * 100.0 for k in keys]
        colors = [EXPERT_COLORS.get(k, "#6b7280") for k in keys]
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
        title_suffix = " · Holiday Gate Active (100% Covariate-Native)" if known_cov is not None else ""
        w_fig.update_layout(
            title=f"Live Causal Pop-Quiz Posterior Weights{title_suffix}",
            template="plotly_white",
            height=220,
            margin=dict(l=20, r=45, t=45, b=20),
            xaxis=dict(range=[0, 110], title="Linear-Pool Mixture-CDF Weight (%)"),
            yaxis=dict(autorange="reversed"),
        )

    out_df = pd.DataFrame(
        {
            "timestamp": future_x,
            "q10": np.round(q10, 4),
            "median": np.round(med, 4),
            "q90": np.round(q90, 4),
        }
    )
    return fig, w_fig, "\n\n".join(metrics_lines), out_df


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), title="Zitra v3 Live Demo (Apache-2.0)") as demo:
    gr.Markdown(
        """
        # ⚡ Zitra v3 — 100% Apache-2.0 Test-Time Adaptive Time-Series Foundation Model
        **Zero commercial restrictions (Apache-2.0)** · **4 orthogonal architectures (Patch Decoder + T5 Encoder + 30L Conformer + T5 4-MoE)** · **Built-in Public Holidays for 240+ countries**

        Pick a preset or upload any CSV to watch **Zitra-Titan-v3** quiz all four Apache-2.0 foundation priors on your series in real time and invert their exact **Linear-Pool Mixture CDF**. Toggle **Public Holiday Covariates** on the Grocery preset to watch the router automatically shift 100% onto covariate-native experts!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            preset_name = gr.Dropdown(
                choices=[
                    "Grocery Daily Orders (Czech Public Holidays + Weekly Cycle)",
                    "Fulfillment Center Hourly Pick Volume",
                    "Solar PV Generation (Hourly Diurnal + Cloud Shifts)",
                    "Cloud CPU Utilization (Regime Shift + Burstiness)",
                    "City Bike Counts (Integer Nonnegative)",
                ],
                value="Grocery Daily Orders (Czech Public Holidays + Weekly Cycle)",
                label="1-Click Sample Series",
            )
            csv_file = gr.File(label="Or Upload Your Own CSV (timestamp + numeric target)", file_types=[".csv"])
            country_code = gr.Dropdown(
                choices=COUNTRY_CHOICES,
                value="CZ",
                label="Built-In Public Holiday Covariates (Optional)",
            )
            horizon = gr.Slider(minimum=7, maximum=168, step=7, value=28, label="Forecast Horizon (steps)")
            eval_holdout = gr.Checkbox(value=True, label="Hold out last H points to score live accuracy")
            compare_stock_t0 = gr.Checkbox(value=True, label="Overlay Stock t0-gamma (Apache-2.0) baseline")
            show_experts = gr.Checkbox(value=True, label="Show individual Zitra-v3 expert medians")
            run_btn = gr.Button("⚡ Forecast Now", variant="primary", size="lg")

        with gr.Column(scale=2):
            summary_md = gr.Markdown("Click **Forecast Now** to run live inference.")
            main_plot = gr.Plot(label="Probabilistic Forecast")
            weights_plot = gr.Plot(label="Live Expert Routing Weights")
            with gr.Accordion("Forecast Table (Downloadable Quantiles)", open=False):
                out_table = gr.Dataframe(interactive=False)

    run_btn.click(
        fn=run_demo,
        inputs=[preset_name, csv_file, country_code, horizon, eval_holdout, compare_stock_t0, show_experts],
        outputs=[main_plot, weights_plot, summary_md, out_table],
    )

if __name__ == "__main__":
    demo.launch()
