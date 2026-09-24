"""
Figure Type 3: Chlorophyll vs CDOM, Inside Upwelling (left) vs Outside Upwelling
(right), pooled across all years. "Inside Upwelling" is now determined by the
trained isolation-forest model (see common.py) instead of a
fixed SST-below-monthly-mean-minus-std threshold.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from common import build_all_data, FIGS_ROOT

OUTPUT_DIR = os.path.join(FIGS_ROOT, "chlor_vs_cdom_upwelling_zones_pooled")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POINTS_POOLED = 50_000


def plot_zones(data_subset, filename_suffix):
    zone_in = data_subset[data_subset["Zone"] == "Inside Upwelling"]
    zone_out = data_subset[data_subset["Zone"] == "Outside Upwelling"]

    if len(zone_in) > MAX_POINTS_POOLED:
        zone_in = zone_in.sample(n=MAX_POINTS_POOLED, random_state=42)
    if len(zone_out) > MAX_POINTS_POOLED:
        zone_out = zone_out.sample(n=MAX_POINTS_POOLED, random_state=42)

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
    plt.savefig(os.path.join(OUTPUT_DIR, f"chlor_vs_cdom_{filename_suffix}.png"), dpi=300, bbox_inches="tight")


def main():
    all_data = build_all_data(use_iforest=True)
    print(f"Total points processed: {len(all_data)}")

    plot_zones(all_data, "all_years_all_seasons")
    plot_zones(all_data[all_data["Season"] == "Upwelling Months (May-Sep)"], "all_years_upwelling_months")
    plot_zones(all_data[all_data["Season"] == "Non-upwelling Months (Oct-Apr)"], "all_years_nonupwelling_months")


if __name__ == "__main__":
    main()