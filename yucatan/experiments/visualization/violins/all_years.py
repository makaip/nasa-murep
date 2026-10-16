import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayth_classification import DATA_PATH, PICKLE_PATH, load_preclassified

# ----------------------------------------------------------------------------
# Paths / params (identical to the IF scatterplot script)
# ----------------------------------------------------------------------------
DATA_PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl"
PRECLASSIFIED = load_preclassified(PICKLE_PATH)

# Upwelling zone bounding box (Yucatan shelf)
UPWELLING_LON_MIN, UPWELLING_LON_MAX = -92.20216, -85.61256
UPWELLING_LAT_MIN, UPWELLING_LAT_MAX = 19.57871, 22.86906

# The old violin script used the whole grid; the IF pipeline restricts to the bbox.
# Set to False to use the full grid instead.
USE_BBOX = True


def build_bbox_mask(metadata):
    lon_centers = np.array(metadata["lon_centers"])
    lat_centers = np.array(metadata["lat_centers"])
    lon_grid_2d, lat_grid_2d = np.meshgrid(lon_centers, lat_centers)
    return (
        (lon_grid_2d >= UPWELLING_LON_MIN) & (lon_grid_2d <= UPWELLING_LON_MAX)
        & (lat_grid_2d >= UPWELLING_LAT_MIN) & (lat_grid_2d <= UPWELLING_LAT_MAX)
    )


# ----------------------------------------------------------------------------
# Load data (v4: has ssl + bathymetry)
# ----------------------------------------------------------------------------
try:
    with open(DATA_PATH, "rb") as f:
        blob = pickle.load(f)
    df = blob["data"]
    metadata = blob["metadata"]
    print("Data loaded successfully.")
except FileNotFoundError:
    raise SystemExit("Data file not found. Please check path.")

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

df = df[(df.index.year >= 2010) & (df.index.year <= 2019)]

grid_shape = np.asarray(df["sst"].iloc[0]).shape


def _shape_ok(a):
    return a is not None and np.asarray(a).shape == grid_shape


df = df[
    df["sst"].apply(_shape_ok)
    & df["ssl"].apply(_shape_ok)
    & df["chlorophyll"].apply(_shape_ok)
    & df["cdom_412"].apply(_shape_ok)
]

region_flat = (build_bbox_mask(metadata) if USE_BBOX else np.ones(grid_shape, dtype=bool)).flatten()


def _is_missing_grid(g):
    return g is None or (isinstance(g, float) and np.isnan(g))


# ----------------------------------------------------------------------------
# Per-year extraction: IF classification is done PER DAY
# ----------------------------------------------------------------------------
def extract_year_data(year, df):
    year_str = str(year)
    print(f"\n{'='*60}\nExtracting Year: {year_str}\n{'='*60}")

    df_year = df[df.index.year == int(year)]
    if df_year.empty:
        print(f"No data for year {year_str}")
        return None

    cdom_labels, cdom_up, s275_all, s300_all, a412_all = [], [], [], [], []
    chl_labels, chl_up, chl_all = [], [], []

    for date_idx, row in df_year.iterrows():
        m = date_idx.month

        month_class = PRECLASSIFIED['by_year'].get(int(year), {}).get(m)
        if month_class is None:
            continue
        up_flat = (month_class['classes'] == 1).flatten()

        s275 = row.get('cdom_slope_275_295')
        s300 = row.get('cdom_slope_300_600')
        a412 = row.get('cdom_412', row.get('cdom'))
        chlor = row.get('chlorophyll', row.get('chlor_a'))

        # --- CDOM ---
        if not any(_is_missing_grid(g) for g in [s275, s300, a412]):
            s275_flat = np.array(s275, dtype=float).flatten()
            s300_flat = np.array(s300, dtype=float).flatten()
            a412_flat = np.array(a412, dtype=float).flatten()

            valid = (
                region_flat
                & ~np.isnan(s275_flat) & ~np.isnan(s300_flat)
                & ~np.isnan(a412_flat) & (a412_flat > 0)
            )
            if np.any(valid):
                cdom_labels.extend([m] * int(valid.sum()))
                cdom_up.extend(up_flat[valid])
                s275_all.extend(s275_flat[valid])
                s300_all.extend(s300_flat[valid])
                a412_all.extend(a412_flat[valid])

        # --- Chlorophyll ---
        if not _is_missing_grid(chlor):
            chlor_flat = np.array(chlor, dtype=float).flatten()
            valid = region_flat & ~np.isnan(chlor_flat) & (chlor_flat > 0)
            if np.any(valid):
                chl_labels.extend([m] * int(valid.sum()))
                chl_up.extend(up_flat[valid])
                chl_all.extend(np.log10(chlor_flat[valid]))

    cdom_labels = np.array(cdom_labels, dtype=int)
    chl_labels = np.array(chl_labels, dtype=int)
    chl_all = np.array(chl_all)

    if cdom_labels.size == 0 and chl_all.size == 0:
        print(f"No valid CDOM or Chlorophyll data found for {year_str}.")
        return None

    return {
        'year': year_str,
        'labels': cdom_labels,
        'is_upwelling': np.array(cdom_up, dtype=bool),
        's275': np.array(s275_all),
        's300': np.array(s300_all),
        'a412': np.array(a412_all),
        'chlor_labels': chl_labels,
        'chlor_is_upwelling': np.array(chl_up, dtype=bool),
        'chlor': chl_all,
    }


def get_trimmed_limits(data_list):
    """Absolute min/max of all data after 3-sigma trimming (per month, per group)."""
    all_trimmed = []
    for month_data in data_list:
        labels = month_data['labels']
        is_up = month_data['is_upwelling']
        values = month_data['values']
        for m in range(1, 13):
            month_mask = labels == m
            up_vals = values[month_mask & is_up]
            non_vals = values[month_mask & ~is_up]
            up_vals = up_vals[~np.isnan(up_vals)]
            non_vals = non_vals[~np.isnan(non_vals)]

            if up_vals.size > 1:
                u_m, u_s = np.mean(up_vals), np.std(up_vals)
                all_trimmed.extend(up_vals[(up_vals >= u_m - 3*u_s) & (up_vals <= u_m + 3*u_s)])
            if non_vals.size > 1:
                n_m, n_s = np.mean(non_vals), np.std(non_vals)
                all_trimmed.extend(non_vals[(non_vals >= n_m - 3*n_s) & (non_vals <= n_m + 3*n_s)])

    if not all_trimmed:
        return 0, 1
    return np.min(all_trimmed), np.max(all_trimmed)


def plot_violin(year_str, values, labels, is_upwelling, var_title, y_unit, y_lims):
    if values.size == 0:
        print(f"No data for {var_title} in {year_str}")
        return None

    fig, ax = plt.subplots(figsize=(16, 6))
    up_data, non_data = [], []
    up_positions, non_positions = [], []

    for month in range(1, 13):
        month_mask = labels == month
        up_vals = values[month_mask & is_upwelling]
        non_vals = values[month_mask & ~is_upwelling]
        up_vals = up_vals[~np.isnan(up_vals)]
        non_vals = non_vals[~np.isnan(non_vals)]

        if up_vals.size > 1:
            mu, sd = np.mean(up_vals), np.std(up_vals)
            up_vals = up_vals[(up_vals >= mu - 3*sd) & (up_vals <= mu + 3*sd)]
        if non_vals.size > 1:
            mu, sd = np.mean(non_vals), np.std(non_vals)
            non_vals = non_vals[(non_vals >= mu - 3*sd) & (non_vals <= mu + 3*sd)]

        if up_vals.size > 1:
            up_data.append(up_vals)
            up_positions.append(month - 0.2)
        if non_vals.size > 1:
            non_data.append(non_vals)
            non_positions.append(month + 0.2)

    for dataset, positions, color in ((up_data, up_positions, 'red'), (non_data, non_positions, 'blue')):
        if not dataset:
            continue
        vp = ax.violinplot(dataset, positions=positions, widths=0.35,
                           showmeans=False, showmedians=True, showextrema=True)
        for pc in vp['bodies']:
            pc.set_facecolor(color); pc.set_edgecolor('black'); pc.set_alpha(0.60)
        for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp[part].set_edgecolor('black'); vp[part].set_linewidth(1.2)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_ylabel(y_unit)

    y_min, y_max = y_lims
    buffer = (y_max - y_min) * 0.05
    new_bottom, new_top = y_min - buffer, y_max + buffer
    if 'Log10' not in var_title:
        new_bottom = max(0, new_bottom)
    ax.set_ylim(bottom=new_bottom, top=new_top)
    ax.grid(True, alpha=0.25, linestyle='--')

    ax.legend(handles=[
        Patch(facecolor='red', edgecolor='black', alpha=0.60, label='Upwelling'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.60, label='Non-upwelling'),
    ], loc='best')

    plt.tight_layout()
    return fig, f"{var_title.replace(':', '_').replace(' ', '_')}_{year_str}.png"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
all_yearly_data = []
for year in range(2010, 2020):
    yd = extract_year_data(year, df)
    if yd:
        all_yearly_data.append(yd)

s275_list = [{'labels': yd['labels'], 'is_upwelling': yd['is_upwelling'], 'values': yd['s275']} for yd in all_yearly_data]
s300_list = [{'labels': yd['labels'], 'is_upwelling': yd['is_upwelling'], 'values': yd['s300']} for yd in all_yearly_data]
a412_list = [{'labels': yd['labels'], 'is_upwelling': yd['is_upwelling'], 'values': yd['a412']} for yd in all_yearly_data]
chlor_list = [{'labels': yd['chlor_labels'], 'is_upwelling': yd['chlor_is_upwelling'], 'values': yd['chlor']} for yd in all_yearly_data]

limits = {
    's275': get_trimmed_limits(s275_list),
    's300': get_trimmed_limits(s300_list),
    'a412': get_trimmed_limits(a412_list),
    'chlor': get_trimmed_limits(chlor_list),
}
print(f"Global limits computed:\nS275: {limits['s275']}\nS300: {limits['s300']}\n"
      f"a412: {limits['a412']}\nChlor: {limits['chlor']}")

all_figures = []
for yd in all_yearly_data:
    ys = yd['year']
    print(f"Generating plots for {ys}...")
    for res in (
        plot_violin(ys, yd['s275'], yd['labels'], yd['is_upwelling'], 'CDOM S275:295',
                    r'$S_{275-295}$ $\text{ (nm}^{-1}\text{)}$', limits['s275']),
        plot_violin(ys, yd['s300'], yd['labels'], yd['is_upwelling'], 'CDOM S300:600',
                    r'$S_{300-600}$ $\text{ (nm}^{-1}\text{)}$', limits['s300']),
        plot_violin(ys, yd['a412'], yd['labels'], yd['is_upwelling'], 'CDOM a(412)',
                    r'$a_{\text{CDOM}}$ 412 $\text{ (m}^{-1}\text{)}$', limits['a412']),
        plot_violin(ys, yd['chlor'], yd['chlor_labels'], yd['chlor_is_upwelling'], 'Log10 Chlorophyll',
                    r'$\log_{10}(\text{mg/m}^3)$', limits['chlor']),
    ):
        if res:
            all_figures.append(res)

FIGS_DIR = Path(__file__).resolve().parents[3] / 'figs' / 'violins'
FIGS_DIR.mkdir(parents=True, exist_ok=True)
for fig, name in all_figures:
    fig.savefig(FIGS_DIR / name, bbox_inches='tight', dpi=300)
    plt.close(fig)

print(f"Saved {len(all_figures)} figures to figs/ directory.")