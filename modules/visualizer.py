import io
import base64
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = ["#7C3AED", "#06B6D4", "#F59E0B", "#10B981"]
BG = "#0F172A"
PANEL = "#1E293B"
TEXT = "#E2E8F0"
GRID = "#334155"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
    }
)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Individual Charts ─────────────────────────────────────────────────────────

def chart_segment_pie(rfm_df):
    counts = rfm_df["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=COLORS[: len(counts)],
        startangle=140,
        wedgeprops=dict(linewidth=2, edgecolor=BG),
    )
    for t in texts + autotexts:
        t.set_color(TEXT)
        t.set_fontsize(10)
    ax.set_title("Customer Segment Distribution", color=TEXT, fontsize=13, pad=14)
    return _fig_to_b64(fig)


def chart_recency_hist(rfm_df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rfm_df["Recency"], bins=40, color=COLORS[0], edgecolor=BG, alpha=0.9)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Recency Distribution", color=TEXT, fontsize=13, pad=14)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    return _fig_to_b64(fig)


def chart_freq_monetary_scatter(rfm_df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(
        rfm_df["Frequency"],
        rfm_df["Monetary"],
        c=rfm_df["Frequency"],
        cmap="plasma",
        alpha=0.6,
        s=15,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Frequency")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Monetary (£)")
    ax.set_title("Frequency vs Monetary", color=TEXT, fontsize=13, pad=14)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    return _fig_to_b64(fig)


def chart_recency_freq_scatter(rfm_df):
    seg_list = rfm_df["Segment"].unique()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, seg in enumerate(seg_list):
        sub = rfm_df[rfm_df["Segment"] == seg]
        ax.scatter(sub["Recency"], sub["Frequency"],
                   label=seg, alpha=0.6, s=15,
                   color=COLORS[i % len(COLORS)], edgecolors="none")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Recency vs Frequency by Segment", color=TEXT, fontsize=13, pad=14)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    return _fig_to_b64(fig)


def chart_top_customers(rfm_df, n=10):
    top = rfm_df.nlargest(n, "Monetary")[["Customer ID", "Monetary", "Segment"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        top["Customer ID"].astype(str),
        top["Monetary"],
        color=COLORS[2],
        edgecolor=BG,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Total Spend (£)")
    ax.set_title(f"Top {n} Customers by Revenue", color=TEXT, fontsize=13, pad=14)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width * 1.01, bar.get_y() + bar.get_height() / 2,
            f"£{width:,.0f}", va="center", fontsize=8, color=TEXT,
        )
    return _fig_to_b64(fig)


def chart_sales_prediction(sales_data):
    hist = sales_data["historical"]
    fore = sales_data["forecast"]

    hist_months = [d["month"] for d in hist]
    hist_sales = [d["sales"] for d in hist]
    fore_months = [d["month"] for d in fore]
    fore_sales = [d["sales"] for d in fore]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_months, hist_sales, color=COLORS[1], linewidth=2, label="Historical")
    # Connect last historical point to first forecast
    ax.plot(
        [hist_months[-1]] + fore_months,
        [hist_sales[-1]] + fore_sales,
        color=COLORS[2],
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=6,
        label="Forecast",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales (£)")
    ax.set_title("Monthly Sales & Forecast (Linear Regression)", color=TEXT, fontsize=13, pad=14)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # Show only a subset of x-labels to avoid clutter
    tick_every = max(1, len(hist_months) // 10)
    ax.set_xticks(hist_months[::tick_every])
    ax.set_xticklabels(hist_months[::tick_every], rotation=45, ha="right", fontsize=8)
    return _fig_to_b64(fig)


# ── Master generator ──────────────────────────────────────────────────────────

def generate_all_charts(rfm_df, sales_data):
    return {
        "pie":          chart_segment_pie(rfm_df),
        "recency_hist": chart_recency_hist(rfm_df),
        "freq_mon":     chart_freq_monetary_scatter(rfm_df),
        "rec_freq":     chart_recency_freq_scatter(rfm_df),
        "top_customers": chart_top_customers(rfm_df),
        "sales":        chart_sales_prediction(sales_data),
    }
