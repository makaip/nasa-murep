"""
Figure Type 4: Chlorophyll vs CDOM, Inside Upwelling (left) vs Outside Upwelling
(right) during Upwelling Months (May-Sep), one figure per year. "Inside Upwelling"
is determined by the trained isolation-forest model (see common.py).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from common import build_all_data, FIGS_ROOT

OUTPUT_DIR = os.path.join(FIGS_ROOT, "chlor_vs_cdom_upwelling_zones_yearly")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POINTS = 50_000


def main():
    all_data = build_all_data(use_iforest=True)
    print(f"Total points processed: {len(all_data)}")

    years = sorted(all_data["year"].unique())

    for year in years:
        year_data = all_data[
            (all_data["year"] == year) & (all_data["Season"] == "Upwelling Months (May-Sep)")
        ].copy()
        if year_data.empty:
            continue

        zone_in = year_data[year_data["Zone"] == "Inside Upwelling"]
        zone_out = year_data[year_data["Zone"] == "Outside Upwelling"]

        if len(zone_in) > MAX_POINTS:
            zone_in = zone_in.sample(n=MAX_POINTS, random_state=42)
        if len(zone_out) > MAX_POINTS:
            zone_out = zone_out.sample(n=MAX_POINTS, random_state=42)

        y_min = min(zone_in["chlor"].min(), zone_out["chlor"].min())

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

        sns.scatterplot(
            ax=axes[0], data=zone_in, x="cdom412", y="chlor",
            color="red", alpha=0.8, marker="o", s=20, linewidth=0.2, edgecolor="black",
        )
        axes[0].set_title("Inside Upwelling")
        axes[0].set_xlabel(r"$a_{\text{CDOM}}$ 412 $(m^{-1})$")
        axes[0].set_ylabel("Chlorophyll-a")
        axes[0].set_yscale("log")
        axes[0].set_xlim(left=0)
        axes[0].set_ylim(bottom=y_min)

        sns.scatterplot(
            ax=axes[1], data=zone_out, x="cdom412", y="chlor",
            color="blue", alpha=0.8, marker="o", s=20, linewidth=0.2, edgecolor="black",
        )
        axes[1].set_title("Outside Upwelling")
        axes[1].set_xlabel(r"$a_{\text{CDOM}}$ 412 $(m^{-1})$")
        axes[1].set_yscale("log")
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(bottom=y_min)

        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"chlor_vs_cdom_upwelling_months_{year}.png"),
            dpi=300, bbox_inches="tight",
        )


if __name__ == "__main__":
    main()