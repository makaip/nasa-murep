"""
Figure Type 1: Chlorophyll vs CDOM, Upwelling Months (May-Sep) vs Non-upwelling
Months (Oct-Apr), one figure per year (2010-2019).

This split is purely calendar-based (Season), so it does not need the isolation-
forest model - we load with use_iforest=False for speed.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from common import build_all_data, FIGS_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_nice_xaxis, nice_log_limits

OUTPUT_DIR = os.path.join(FIGS_ROOT, "chlor_vs_cdom_upwelling_months_2010_2019")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POINTS = 50_000


def main():
    all_data = build_all_data(use_iforest=False)
    print(f"Total points processed: {len(all_data)}")

    years = sorted(all_data["year"].unique())

    for year in years:
        year_data = all_data[all_data["year"] == year].copy()
        if year_data.empty:
            continue

        upwelling_pts = year_data[year_data["Season"] == "Upwelling Months (May-Sep)"]
        nonupwelling_pts = year_data[year_data["Season"] == "Non-upwelling Months (Oct-Apr)"]

        if len(upwelling_pts) > MAX_POINTS:
            upwelling_pts = upwelling_pts.sample(n=MAX_POINTS, random_state=42)
        if len(nonupwelling_pts) > MAX_POINTS:
            nonupwelling_pts = nonupwelling_pts.sample(n=MAX_POINTS, random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

        sns.scatterplot(
            ax=axes[0], data=upwelling_pts, x="cdom412", y="chlor",
            color="red", alpha=0.8, marker="o", s=20, linewidth=0.2, edgecolor="black",
        )
        axes[0].set_title("Upwelling Months (May-Sep)")
        axes[0].set_xlabel(r"$a_{\text{CDOM}}$ 412 $(m^{-1})$")
        axes[0].set_ylabel("Chlorophyll-a")
        axes[0].set_yscale("log")

        sns.scatterplot(
            ax=axes[1], data=nonupwelling_pts, x="cdom412", y="chlor",
            color="blue", alpha=0.8, marker="o", s=20, linewidth=0.2, edgecolor="black",
        )
        axes[1].set_title("Non-upwelling Months (Oct-Apr)")
        axes[1].set_xlabel(r"$a_{\text{CDOM}}$ 412 $(m^{-1})$")
        axes[1].set_yscale("log")

        # Shared axes: round limits outward to clean values. Negative aCDOM values are kept.
        apply_nice_xaxis(axes[0], min(upwelling_pts["cdom412"].min(), nonupwelling_pts["cdom412"].min()),
                         max(upwelling_pts["cdom412"].max(), nonupwelling_pts["cdom412"].max()))
        y_limits = nice_log_limits(min(upwelling_pts["chlor"].min(), nonupwelling_pts["chlor"].min()),
                                   max(upwelling_pts["chlor"].max(), nonupwelling_pts["chlor"].max()))
        if y_limits is not None:
            axes[0].set_ylim(*y_limits)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"chlor_vs_cdom_{year}.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()