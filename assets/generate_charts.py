"""Generate crisp publication-grade charts for Zitra v2 Hugging Face cards & README."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#e5e7eb",
    "axes.linewidth": 1.0,
    "grid.color": "#f3f4f6",
    "grid.linestyle": "--",
})


def make_benchmark_chart():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220)
    fig.patch.set_facecolor("white")

    # Left panel: GIFT-Eval (Lower is Better!)
    models_g = [
        ("Chronos-2", 0.698, 0.485, "#94a3b8", "o", (8, -4), False),
        ("Toto-2.0-2.5B", 0.696, 0.476, "#94a3b8", "o", (8, -4), False),
        ("TimesFM-3", 0.667, 0.456, "#64748b", "s", (8, -4), False),
        ("t0-gamma", 0.661, 0.452, "#f59e0b", "D", (8, -4), False),
        ("Timestral-209M", 0.6575, 0.4460, "#ef4444", "s", (9, 6), False),
        ("Zitra-209M-v2 (1.0x)", 0.6578, 0.4451, "#6366f1", "^", (9, -11), False),
        ("Zitra-Titan-585M", 0.6443, 0.4352, "#4f46e5", "*", (11, -3), False),
    ]
    ax = axes[0]
    ax.set_facecolor("#fafafa")
    ax.grid(True, zorder=0)
    for name, mase, crps, col, mk, off, _ in models_g:
        sz = 260 if "Titan" in name else (110 if "Zitra" in name or "Timestral" in name else 75)
        ax.scatter(mase, crps, c=col, marker=mk, s=sz, zorder=4, edgecolors="white", linewidths=1.2)
        weight = "bold" if "Zitra-Titan" in name or "Timestral" in name else "normal"
        ax.annotate(
            name,
            (mase, crps),
            xytext=off,
            textcoords="offset points",
            fontsize=9.5 if "Titan" in name else 8.8,
            fontweight=weight,
            color="#111827" if "Zitra" in name or "Timestral" in name else "#475569",
        )
    # Arrow highlighting the jump from Timestral-209M to Zitra-Titan-585M
    ax.annotate(
        "",
        xy=(0.6451, 0.4359),
        xytext=(0.6568, 0.4454),
        arrowprops=dict(arrowstyle="->", color="#4f46e5", lw=2.0, ls="--"),
        zorder=3,
    )
    ax.text(
        0.6416, 0.4472,
        "−2.02% MASE\n−2.43% CRPS",
        color="#4f46e5",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c7d2fe", alpha=0.95),
    )
    ax.set_title("Official GIFT-Eval (97 Configs) — Lower is Better ↓", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Normalized Geometric Mean MASE ↓", fontsize=10)
    ax.set_ylabel("Normalized Geometric Mean CRPS / WQL ↓", fontsize=10)
    ax.set_xlim(0.640, 0.715)
    ax.set_ylim(0.431, 0.491)

    # Right panel: FEV-Bench (100 Tasks — Higher Skill Score is Better!)
    models_f = [
        ("Toto-2.0-2.5B", 32.54, 44.42, "#94a3b8", "o", (33.1, 44.35), False),
        ("Chronos-2", 35.50, 47.28, "#94a3b8", "o", (36.0, 47.20), False),
        ("TimesFM-3", 37.82, 48.99, "#64748b", "s", (38.3, 48.85), False),
        ("t0-gamma", 37.99, 49.42, "#f59e0b", "D", (36.2, 49.45), False),
        ("Timestral-209M", 39.49, 50.39, "#ef4444", "s", (36.3, 50.25), True),
        ("Zitra-209M-v2", 39.56, 50.45, "#6366f1", "^", (38.85, 49.70), True),
        ("Zitra-Titan-585M", 39.71, 50.61, "#4f46e5", "*", (36.6, 50.95), True),
        ("Zitra-Apex-585M", 39.85, 50.55, "#10b981", "P", (38.85, 50.10), True),
    ]
    ax = axes[1]
    ax.set_facecolor("#fafafa")
    ax.grid(True, zorder=0)
    for name, mase_s, sql_s, col, mk, xy_txt, use_arrow in models_f:
        sz = 260 if "Titan" in name else (120 if "Zitra" in name or "Timestral" in name else 75)
        ax.scatter(mase_s, sql_s, c=col, marker=mk, s=sz, zorder=4, edgecolors="white", linewidths=1.2)
        weight = "bold" if "Zitra" in name or "Timestral" in name else "normal"
        ap = dict(arrowstyle="-", color="#94a3b8", lw=0.9) if use_arrow else None
        ax.annotate(
            name,
            xy=(mase_s, sql_s),
            xytext=xy_txt,
            fontsize=9.1 if "Zitra" in name else 8.8,
            fontweight=weight,
            color="#111827" if "Zitra" in name or "Timestral" in name else "#475569",
            arrowprops=ap,
            zorder=5,
        )
    ax.set_title("Official fev-bench (100 Tasks) — Higher Skill is Better ↑", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("MASE Skill Score (%) ↑", fontsize=10)
    ax.set_ylabel("SQL Skill Score (%) ↑", fontsize=10)
    ax.set_xlim(31.8, 41.3)
    ax.set_ylim(43.8, 51.45)

    fig.suptitle("Zitra v2 Sets a New State of the Art on Both GIFT-Eval and fev-bench", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = OUT / "pareto_gift_fev.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


def make_rohlik_chart():
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", zorder=0)

    categories = [
        "Public Rohlik\nOrders (FEV)",
        "Public Rohlik\nSales (FEV)",
        "Internal 2026\nHourly FC Orders\n(H=24h)",
        "Internal 2026\nHourly FC Orders\n(H=168h)",
        "Internal 2026\nHourly Picked\n(H=168h)",
        "Internal 2026\n5,000-SKU Sales\n(H=14d)",
    ]
    # WQL reduction (%) vs Stock Timestral-209M (positive bars = % error reduction!)
    titan_vals = [3.92, 7.21, 2.11, 2.08, 0.98, 0.99]
    apex_vals = [2.40, 7.14, 3.83, 2.57, 1.93, 0.76]
    z209_vals = [0.14, 0.31, 0.34, -0.15, -0.23, 0.68]

    x = np.arange(len(categories))
    w = 0.26
    b1 = ax.bar(x - w, titan_vals, w, label="Zitra-Titan-585M (Flagship)", color="#4f46e5", zorder=3)
    b2 = ax.bar(x, apex_vals, w, label="Zitra-Apex-585M (Fast 3-Prior)", color="#10b981", zorder=3)
    b3 = ax.bar(x + w, z209_vals, w, label="Zitra-209M-v2 (1.00x Speed)", color="#818cf8", zorder=3)

    ax.axhline(0, color="#374151", linewidth=1.0, zorder=2)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.14,
                f"−{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color="#1f2937",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("WQL Error Reduction vs Timestral-209M (%) ↑", fontsize=10)
    ax.set_title("Zero-Shot Retail & Fulfillment Performance on Rohlik Benchmarks (Higher Bar = Lower Error)", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_ylim(-0.6, 8.4)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    p = OUT / "rohlik_benchmark.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Saved", p)


if __name__ == "__main__":
    make_benchmark_chart()
    make_rohlik_chart()
