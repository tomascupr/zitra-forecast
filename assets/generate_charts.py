"""Generate publication-grade, clutter-free benchmark charts for Zitra v3 (Apache-2.0)."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.9,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.65,
})


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, zorder=0)


def make_benchmark_chart():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=220)
    fig.patch.set_facecolor("white")

    # Left panel: Official GIFT-Eval (Lower is better)
    ax = axes[0]
    _style_ax(ax)

    # (label, mase, crps, color, marker, size, xytext_pts, bold, use_arrow)
    models_g = [
        ("Chronos-2",      0.6978, 0.4854, "#10b981", "o", 85,  (-6, -18), False, False),
        ("Toto-2.0-2.5B",  0.6956, 0.4759, "#10b981", "o", 85,  (-18, -18), False, False),
        ("TimesFM-3",      0.6668, 0.4557, "#94a3b8", "s", 85,  (9, -3),   False, False),
        ("t0-gamma",       0.6608, 0.4520, "#10b981", "o", 105, (10, -4),  False, False),
        ("Timestral-209M", 0.6575, 0.4460, "#94a3b8", "s", 105, (10, -4),  True,  False),
        ("Zitra-Core-v3",  0.6551, 0.4473, "#0ea5e9", "^", 145, (-26, 26), True,  True),
        ("Zitra-Titan-v3", 0.6523, 0.4420, "#4f46e5", "*", 380, (12, -5),  True,  False),
    ]
    for name, mase, crps, col, mk, sz, off, bold, arr in models_g:
        ec = "#1e1b4b" if "Titan" in name else "white"
        ax.scatter(mase, crps, c=col, marker=mk, s=sz, zorder=5 if bold else 4, edgecolors=ec, linewidths=1.1)
        color_txt = "#3730a3" if "Titan" in name else ("#0284c7" if "Core" in name else ("#1e293b" if bold else "#475569"))
        ax.annotate(
            name,
            xy=(mase, crps),
            xytext=off,
            textcoords="offset points",
            fontsize=10.0 if "Titan" in name else (9.0 if bold else 8.8),
            fontweight="bold" if bold else "normal",
            color=color_txt,
            arrowprops=dict(arrowstyle="-", color="#0ea5e9", lw=1.0) if arr else None,
            zorder=6,
        )

    ax.annotate(
        "",
        xy=(0.6532, 0.4426),
        xytext=(0.6601, 0.4512),
        arrowprops=dict(arrowstyle="->", color="#4f46e5", lw=1.9, ls="--"),
        zorder=3,
    )
    ax.text(
        0.6468, 0.4805,
        "Better ← lower error\n−1.3% MASE / −2.2% CRPS vs. t0-gamma\n−0.8% MASE / −0.9% CRPS vs. Timestral-209M",
        color="#312e81",
        fontsize=8.4,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#eef2ff", edgecolor="#c7d2fe", alpha=0.96),
        zorder=6,
    )
    ax.set_title("Official GIFT-Eval (Lower ↓)", fontsize=12.0, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel("Normalized Geometric Mean MASE ↓", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("Normalized Geometric Mean CRPS ↓", fontsize=10, fontweight="bold", color="#334155")
    ax.set_xlim(0.645, 0.706)
    ax.set_ylim(0.436, 0.491)

    # Right panel: Official fev-bench (Higher is better)
    ax = axes[1]
    _style_ax(ax)
    ax.axhline(50.0, color="#6366f1", linestyle=":", linewidth=1.2, alpha=0.55, zorder=1)
    ax.text(32.05, 50.08, "50% SQL skill", fontsize=8.0, color="#6366f1", style="italic")

    models_f = [
        ("Toto-2.0-2.5B",  32.54, 44.42, "#10b981", "o", 85,  (33.1, 44.32), False, False),
        ("Chronos-2",      35.50, 47.28, "#10b981", "o", 85,  (36.05, 47.18), False, False),
        ("TimesFM-3",      37.82, 48.99, "#94a3b8", "s", 85,  (35.35, 48.62), False, True),
        ("t0-gamma",       37.99, 49.42, "#10b981", "o", 105, (35.15, 49.25), False, True),
        ("Zitra-Core-v3",  38.39, 49.69, "#0ea5e9", "^", 145, (34.65, 49.72), True,  True),
        ("Zitra-Titan-v3", 39.21, 50.30, "#4f46e5", "*", 380, (34.45, 50.45), True,  True),
        ("Timestral-209M", 39.49, 50.39, "#94a3b8", "s", 105, (35.45, 51.08), True,  True),
    ]
    for name, mase_s, sql_s, col, mk, sz, xy_txt, bold, arr in models_f:
        ec = "#1e1b4b" if "Titan" in name else "white"
        ax.scatter(mase_s, sql_s, c=col, marker=mk, s=sz, zorder=5 if bold else 4, edgecolors=ec, linewidths=1.1)
        color_txt = "#3730a3" if "Titan" in name else ("#0284c7" if "Core" in name else ("#1e293b" if bold else "#475569"))
        ax.annotate(
            name,
            xy=(mase_s, sql_s),
            xytext=xy_txt,
            fontsize=10.0 if "Titan" in name else (9.0 if bold else 8.8),
            fontweight="bold" if bold else "normal",
            color=color_txt,
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.95) if arr else None,
            zorder=6,
        )
    ax.set_title("Official fev-bench (Higher ↑)", fontsize=12.0, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel("MASE Skill Score (%) ↑", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("SQL Skill Score (%) ↑", fontsize=10, fontweight="bold", color="#334155")
    ax.set_xlim(31.8, 40.6)
    ax.set_ylim(43.8, 51.65)

    # Single shared legend below title — eliminates repeated "(Apache-2.0)" clutter
    leg = [
        Line2D([0], [0], marker="*", color="w", label="Zitra-Titan-v3 (Apache-2.0)", markerfacecolor="#4f46e5", markeredgecolor="#1e1b4b", markersize=13),
        Line2D([0], [0], marker="^", color="w", label="Zitra-Core-v3 (Apache-2.0)", markerfacecolor="#0ea5e9", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Other Apache-2.0", markerfacecolor="#10b981", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Non-Commercial", markerfacecolor="#94a3b8", markersize=8),
    ]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False, fontsize=9.2)
    fig.suptitle(
        "Zitra v3 — #1 & #2 Apache-2.0 Models Across GIFT-Eval and fev-bench",
        fontsize=13.6,
        fontweight="bold",
        color="#0f172a",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    p = OUT / "pareto_gift_fev.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


def make_core_benchmarks_chart():
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), dpi=220)
    fig.patch.set_facecolor("white")

    # Left panel: fev-bench Accuracy vs Speed Frontier (log x-axis)
    ax = axes[0]
    _style_ax(ax)

    pts = [
        ("Zitra-Core-v3",  1807,  49.69, "#4f46e5", "*", 420, True,  "core"),
        ("Zitra-Titan-v3", 5719,  50.30, "#0f172a", "D", 130, True,  "titan"),
        ("Timestral-209M", 3827,  50.39, "#94a3b8", "s", 85,  False, "nc"),
        ("t0-gamma",       15862, 49.42, "#10b981", "o", 85,  False, "ap"),
        ("TimesFM-3",      8673,  48.99, "#94a3b8", "s", 85,  False, "nc"),
        ("Chronos-2",      2738,  47.28, "#10b981", "o", 85,  False, "ap"),
        ("iAmTime",        599,   46.78, "#10b981", "o", 85,  False, "ap"),
        ("Toto-2.0-2.5B",  13835, 44.42, "#10b981", "o", 85,  False, "ap"),
    ]
    for name, rt, sql, col, mk, sz, bold, kind in pts:
        ec = "#1e1b4b" if kind == "core" else "white"
        ax.scatter(rt, sql, c=col, marker=mk, s=sz, zorder=5 if bold else 4, edgecolors=ec, linewidths=1.2)

    ax.annotate(
        "Zitra-Core-v3 (256M)\n49.69% · 1,807s",
        xy=(1807, 49.69), xytext=(1120, 50.52),
        ha="center", fontsize=8.8, fontweight="bold", color="#312e81",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef2ff", edgecolor="#a5b4fc", alpha=0.97),
        arrowprops=dict(arrowstyle="-", color="#6366f1", lw=1.0), zorder=6,
    )
    ax.annotate(
        "Zitra-Titan-v3 (1.06B)\n50.30% · 5,719s",
        xy=(5719, 50.30), xytext=(11400, 50.95),
        ha="center", fontsize=8.8, fontweight="bold", color="#0f172a",
        arrowprops=dict(arrowstyle="-", color="#64748b", lw=1.0), zorder=6,
    )
    ax.annotate(
        "Timestral-209M\n50.39% · 3,827s",
        xy=(3827, 50.39), xytext=(3827, 51.08),
        ha="center", fontsize=8.3, color="#475569",
        arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=0.9), zorder=5,
    )
    ax.annotate(
        "t0-gamma\n49.42% · 15,862s",
        xy=(15862, 49.42), xytext=(15500, 48.32),
        ha="center", fontsize=8.3, color="#334155",
        arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=0.9), zorder=5,
    )
    ax.annotate(
        "TimesFM-3\n48.99% · 8,673s",
        xy=(8673, 48.99), xytext=(8200, 47.62),
        ha="center", fontsize=8.3, color="#475569",
        arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=0.9), zorder=5,
    )
    ax.text(2920, 46.82, "Chronos-2\n47.28% · 2,738s", fontsize=8.2, color="#334155")
    ax.text(660, 46.62, "iAmTime\n46.78% · 599s", fontsize=8.2, color="#334155")
    ax.text(11200, 44.32, "Toto-2.0-2.5B\n44.42% · 13,835s", ha="right", fontsize=8.2, color="#334155")

    ax.annotate(
        "",
        xy=(2000, 49.58), xytext=(15200, 49.38),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.22", color="#4f46e5", lw=2.0, ls="--"),
        zorder=3,
    )
    ax.text(
        4800, 48.42,
        "8.8× faster & +0.27 pp SQL\nvs. stock t0-gamma",
        ha="center", fontsize=8.4, fontweight="bold", color="#3730a3",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef2ff", edgecolor="#c7d2fe", alpha=0.95),
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["500s", "1,000s", "2,000s", "5,000s", "10,000s", "20,000s"], fontsize=9)
    ax.set_xlim(420, 24000)
    ax.set_ylim(43.8, 51.85)
    ax.set_xlabel("Total fev-bench Runtime (log scale · faster ←)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_ylabel("fev-bench SQL Skill Score (% · higher ↑)", fontsize=10, fontweight="bold", color="#334155")
    ax.set_title("fev-bench: Accuracy vs. Speed Frontier", fontsize=12, fontweight="bold", pad=16, color="#0f172a")

    leg = [
        Line2D([0], [0], marker="*", color="w", label="Zitra-Core-v3 (Apache-2.0)", markerfacecolor="#4f46e5", markeredgecolor="#1e1b4b", markersize=12),
        Line2D([0], [0], marker="D", color="w", label="Zitra-Titan-v3 (Apache-2.0)", markerfacecolor="#0f172a", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Other Apache-2.0", markerfacecolor="#10b981", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Non-Commercial", markerfacecolor="#94a3b8", markersize=8),
    ]
    ax.legend(handles=leg, loc="lower left", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=8.3)

    # Right panel: Official GIFT-Eval horizontal bars
    ax2 = axes[1]
    _style_ax(ax2)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", linestyle="--", alpha=0.55, zorder=0)

    bars_data = [
        ("Zitra-Titan-v3", 0.6523, 0.4420, "#0f172a", "#475569", True),
        ("Zitra-Core-v3",  0.6551, 0.4473, "#4f46e5", "#818cf8", True),
        ("Timestral-209M", 0.6575, 0.4460, "#94a3b8", "#cbd5e1", False),
        ("t0-gamma",       0.6608, 0.4520, "#10b981", "#6ee7b7", False),
        ("TimesFM-3",      0.6668, 0.4557, "#94a3b8", "#cbd5e1", False),
        ("Toto-2.0-2.5B",  0.6956, 0.4759, "#10b981", "#6ee7b7", False),
        ("Chronos-2",      0.6978, 0.4854, "#10b981", "#6ee7b7", False),
    ]
    y = np.arange(len(bars_data))
    h = 0.36
    for i, (lbl, mase, crps, c_m, c_c, bold) in enumerate(bars_data):
        ax2.barh(i - h / 2, mase, height=h, color=c_m, zorder=3)
        ax2.barh(i + h / 2, crps, height=h, color=c_c, zorder=3)
        txt_col = "#3730a3" if "Core" in lbl else ("#0f172a" if bold else "#334155")
        ax2.text(mase + 0.004, i - h / 2, f"{mase:.4f}", va="center", fontsize=8.8, fontweight="bold" if bold else "normal", color=txt_col)
        ax2.text(crps + 0.004, i + h / 2, f"{crps:.4f}", va="center", fontsize=8.8, fontweight="bold" if bold else "normal", color=txt_col)

    ax2.set_yticks(y)
    ax2.set_yticklabels([b[0] for b in bars_data], fontsize=9.4)
    for tick, b in zip(ax2.get_yticklabels(), bars_data):
        if b[5]:
            tick.set_fontweight("bold")
        if "Core" in b[0]:
            tick.set_color("#3730a3")
    ax2.invert_yaxis()
    ax2.set_xlim(0.40, 0.755)
    ax2.set_xlabel("Normalized Geometric Mean Error (lower ←)", fontsize=10, fontweight="bold", color="#334155")
    ax2.set_title("Official GIFT-Eval: MASE & CRPS (Lower ↓)", fontsize=12, fontweight="bold", pad=26, color="#0f172a")
    ax2.legend(
        handles=[Patch(facecolor="#0f172a", label="MASE ↓"), Patch(facecolor="#475569", label="CRPS ↓")],
        loc="upper right", bbox_to_anchor=(1.0, 1.07), ncol=2, frameon=False, fontsize=8.8,
    )

    fig.suptitle(
        "Zitra-Core-v3 — #1 Compact (256M) Apache-2.0 Forecasting Model",
        fontsize=13.6,
        fontweight="bold",
        color="#0f172a",
        y=1.02,
    )
    plt.tight_layout()
    p = OUT / "zitra_core_v3_benchmarks.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


def make_rohlik_chart():
    fig, ax = plt.subplots(figsize=(12.2, 5.1), dpi=220)
    fig.patch.set_facecolor("white")
    _style_ax(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.55, zorder=0)

    categories = [
        "Kaggle\nOrders 1D",
        "Kaggle\nOrders 1W",
        "Kaggle\nSales 1D",
        "Kaggle\nSales 1W",
        "2026 D+7\nHoliday Wks",
        "2026 Hourly\nOrders (24h)",
        "2026 Hourly\nOrders (7d)",
        "2026 Hourly\nPicked (7d)",
    ]
    # Exact verified % error reduction vs. Timestral-209M (positive = lower error than Timestral-209M)
    titan_vals = [3.88, 5.54, 7.20, 15.42, 25.16, 4.16, 3.66, 0.94]
    core_vals  = [5.98, 5.75, 4.75, 13.90, -0.19, 3.08, 2.90, 0.28]

    x = np.arange(len(categories))
    w = 0.36
    bars_t = ax.bar(x - w / 2, titan_vals, width=w, color="#0f172a", edgecolor="white", linewidth=0.9, label="Zitra-Titan-v3 (1.06B · Apache-2.0)", zorder=3)
    bars_c = ax.bar(x + w / 2, core_vals,  width=w, color="#4f46e5", edgecolor="white", linewidth=0.9, label="Zitra-Core-v3 (256M · Apache-2.0)", zorder=3)
    ax.axhline(0, color="#334155", linewidth=1.0, zorder=4)

    # Subtle vertical dividers & suite headers
    for xv in (3.5, 4.5):
        ax.axvline(xv, color="#cbd5e1", linestyle=":", linewidth=1.1, zorder=1)
    ax.text(1.5, 30.8, "Public Rohlik Kaggle (WQL ↓)", ha="center", va="top", fontsize=8.8, fontweight="bold", color="#475569")
    ax.text(4.0, 30.8, "Built-In Holidays\n(WAPE ↓)", ha="center", va="top", fontsize=8.4, fontweight="bold", color="#059669")
    ax.text(6.0, 30.8, "2026 Hourly Fulfillment Ops (WQL ↓)", ha="center", va="top", fontsize=8.8, fontweight="bold", color="#475569")

    for bars, vals, col in [(bars_t, titan_vals, "#0f172a"), (bars_c, core_vals, "#3730a3")]:
        for bar, v in zip(bars, vals):
            cx = bar.get_x() + bar.get_width() / 2
            if v >= 0:
                ax.text(cx, v + 0.45, f"−{v:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=col)
            else:
                ax.text(cx, v - 0.55, f"+{abs(v):.1f}%", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#64748b")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.0, color="#1e293b")
    ax.set_ylabel("Error Reduction vs. Timestral-209M (%) ↑", fontsize=10, fontweight="bold", color="#334155")
    ax.set_title(
        "Zero-Shot Retail & Fulfillment Gains — Zitra-Titan-v3 & Zitra-Core-v3 vs. Timestral-209M",
        fontsize=12.4,
        fontweight="bold",
        color="#0f172a",
        pad=26,
    )
    ax.set_ylim(-2.6, 32.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, fontsize=9.4)
    plt.tight_layout()
    p = OUT / "rohlik_benchmark.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


if __name__ == "__main__":
    make_benchmark_chart()
    make_core_benchmarks_chart()
    make_rohlik_chart()
