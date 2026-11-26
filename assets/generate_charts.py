"""Generate crisp publication-grade charts for Zitra v3 (100% Apache-2.0) Hugging Face card & README."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d1d5db",
    "axes.linewidth": 1.0,
    "grid.color": "#e5e7eb",
    "grid.linestyle": "--",
})


def make_benchmark_chart():
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=220)
    fig.patch.set_facecolor("white")

    # Left panel: Official GIFT-Eval (97 Configs — Lower is Better!)
    # (name, MASE, CRPS, color, marker, text_offset_pts, bold, is_apache)
    models_g = [
        ("Chronos-2\n(Apache-2.0)", 0.698, 0.485, "#64748b", "o", (-54, -22), False, True),
        ("Toto-2.0-2.5B\n(Apache-2.0)", 0.696, 0.476, "#64748b", "o", (-62, -26), False, True),
        ("TimesFM-3\n(Non-Comm)", 0.667, 0.456, "#94a3b8", "s", (8, -6), False, False),
        ("t0-gamma\n(Prev #1 Apache-2.0)", 0.661, 0.452, "#f59e0b", "D", (10, -14), False, True),
        ("Timestral-209M\n(Non-Commercial)", 0.6575, 0.4460, "#ef4444", "s", (10, -14), True, False),
        ("Zitra-Titan-v3\n(100% Apache-2.0)", 0.6522, 0.4422, "#4f46e5", "*", (12, -14), True, True),
    ]
    ax = axes[0]
    ax.set_facecolor("#fafafa")
    ax.grid(True, zorder=0)
    for name, mase, crps, col, mk, off, bold, _ in models_g:
        sz = 340 if "Zitra" in name else (115 if ("Timestral" in name or "t0-gamma" in name) else 75)
        ax.scatter(mase, crps, c=col, marker=mk, s=sz, zorder=4, edgecolors="white", linewidths=1.2)
        ax.annotate(
            name,
            (mase, crps),
            xytext=off,
            textcoords="offset points",
            fontsize=9.3 if "Zitra" in name else 8.4,
            fontweight="bold" if bold else "normal",
            color="#111827" if (bold or "t0-gamma" in name) else "#475569",
            zorder=5,
        )
    # Arrow from t0-gamma (prev #1 Apache-2.0) to Zitra-Titan-v3
    ax.annotate(
        "",
        xy=(0.6531, 0.4428),
        xytext=(0.6603, 0.4512),
        arrowprops=dict(arrowstyle="->", color="#4f46e5", lw=2.0, ls="--"),
        zorder=3,
    )
    ax.text(
        0.6468, 0.4790,
        "vs t0-gamma: −1.3% MASE / −2.2% CRPS\nvs Timestral: −0.8% MASE / −0.9% CRPS",
        color="#4f46e5",
        fontsize=8.3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#c7d2fe", alpha=0.97),
        zorder=6,
    )
    ax.set_title("Official GIFT-Eval (97 Configs) — Lower is Better ↓", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Normalized Geometric Mean MASE ↓", fontsize=10)
    ax.set_ylabel("Normalized Geometric Mean CRPS / WQL ↓", fontsize=10)
    ax.set_xlim(0.645, 0.708)
    ax.set_ylim(0.436, 0.492)

    # Right panel: Official fev-bench (100 Tasks — Higher Skill Score is Better!)
    models_f = [
        ("Toto-2.0-2.5B (Apache-2.0)", 32.54, 44.42, "#64748b", "o", (33.1, 44.32), False),
        ("Chronos-2 (Apache-2.0)", 35.50, 47.28, "#64748b", "o", (36.0, 47.18), False),
        ("TimesFM-3 (Non-Comm)", 37.82, 48.99, "#94a3b8", "s", (34.4, 48.85), True),
        ("t0-gamma (Prev #1 Apache-2.0)", 37.99, 49.42, "#f59e0b", "D", (33.2, 49.68), True),
        ("Timestral-209M (Non-Commercial)", 39.49, 50.39, "#ef4444", "s", (34.4, 51.02), True),
        ("Zitra-Titan-v3 (100% Apache-2.0)", 39.05, 50.18, "#4f46e5", "*", (33.3, 50.36), True),
    ]
    ax = axes[1]
    ax.set_facecolor("#fafafa")
    ax.grid(True, zorder=0)
    ax.axhline(50.0, color="#6366f1", linestyle=":", linewidth=1.2, alpha=0.6, zorder=1)
    ax.text(32.1, 50.06, "50% SQL Skill Threshold (First Pure Apache-2.0 Model Above 50%)", fontsize=7.8, color="#4f46e5", style="italic")

    for name, mase_s, sql_s, col, mk, xy_txt, use_arrow in models_f:
        sz = 340 if "Zitra" in name else (115 if ("Timestral" in name or "t0-gamma" in name) else 75)
        ax.scatter(mase_s, sql_s, c=col, marker=mk, s=sz, zorder=4, edgecolors="white", linewidths=1.2)
        bold = ("Zitra" in name) or ("Timestral" in name)
        ap = dict(arrowstyle="-", color="#94a3b8", lw=0.9) if use_arrow else None
        ax.annotate(
            name,
            xy=(mase_s, sql_s),
            xytext=xy_txt,
            fontsize=9.1 if "Zitra" in name else 8.4,
            fontweight="bold" if bold else "normal",
            color="#111827" if (bold or "t0-gamma" in name) else "#475569",
            arrowprops=ap,
            zorder=5,
        )
    ax.set_title("Official fev-bench (100 Tasks) — Higher Skill is Better ↑", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("MASE Skill Score (%) ↑", fontsize=10)
    ax.set_ylabel("SQL Skill Score (%) ↑", fontsize=10)
    ax.set_xlim(31.8, 40.8)
    ax.set_ylim(43.8, 51.65)

    fig.suptitle(
        "Zitra v3: #1 Pure Apache-2.0 Time-Series Foundation Model on Both GIFT-Eval and fev-bench",
        fontsize=13.2,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    p = OUT / "pareto_gift_fev.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


def make_rohlik_chart():
    fig, ax = plt.subplots(figsize=(11.2, 4.9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", zorder=0)

    categories = [
        "Public Rohlik\nOrders 1D\n(fev-bench)",
        "Public Rohlik\nOrders 1W\n(fev-bench)",
        "Public Rohlik\nSales 1D\n(fev-bench)",
        "Public Rohlik\nSales 1W\n(fev-bench)",
        "2026 D+7\n31 Holiday Wks\n(w/ Holidays)",
        "2026 Hourly\nFC Orders\n(H=24h)",
        "2026 Hourly\nFC Orders\n(H=168h)",
        "2026 Hourly\nPicked Vol\n(H=168h)",
    ]
    # Exact verified % error reduction of Zitra-Titan-v3 (Pure Apache-2.0) vs Timestral-209M
    # WQL reduction for public fev-bench & hourly ops; WAPE reduction for D+7 31 holiday weeks (7.805% vs 10.393% = -24.9%)
    vals = [4.46, 5.48, 7.07, 15.35, 24.90, 4.00, 3.70, 0.89]
    colors = ["#4f46e5", "#4f46e5", "#4f46e5", "#4f46e5", "#10b981", "#6366f1", "#6366f1", "#6366f1"]

    x = np.arange(len(categories))
    bars = ax.bar(x, vals, width=0.58, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.axhline(0, color="#374151", linewidth=1.0, zorder=2)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.55,
            f"−{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#111827",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.6)
    ax.set_ylabel("Error Reduction vs Timestral-209M (%) ↑", fontsize=10)
    ax.set_title(
        "Zitra-Titan-v3 (100% Apache-2.0) Zero-Shot Retail & Fulfillment Gains vs Timestral-209M",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 29.0)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4f46e5", label="Public Rohlik Kaggle (fev-bench WQL ↓)"),
        Patch(facecolor="#10b981", label="Internal 2026 D+7 Holiday Weeks (Built-in Holidays WAPE ↓)"),
        Patch(facecolor="#6366f1", label="Internal 2026 Hourly FC Operations (No-Covariate WQL ↓)"),
    ]
    ax.legend(handles=legend_handles, frameon=True, facecolor="white", loc="upper left", fontsize=8.5)
    plt.tight_layout()
    p = OUT / "rohlik_benchmark.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


if __name__ == "__main__":
    make_benchmark_chart()
    make_rohlik_chart()
