"""
view raw features by monthly average
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import warnings

plt.style.use('default')

try:
    with open(r'E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v3.pkl', 'rb') as f:
        data = pickle.load(f)

    df = data['data']
    metadata = data['metadata']
    print("Data loaded successfully.")
    
    lon_centers = np.array(metadata['lon_centers'])
    lat_centers = np.array(metadata['lat_centers'])
    if lon_centers.ndim == 1 and lat_centers.ndim == 1:
        lon_grid_2d, lat_grid_2d = np.meshgrid(lon_centers, lat_centers)
    else:
        lon_grid_2d, lat_grid_2d = lon_centers, lat_centers
        
except FileNotFoundError:
    print("Data file not found. Please check path.")


target_year = 2018
months = range(1, 13)
month_data = {
    m: {'sst': [], 'chlor': [], 'ssl': []} for m in months
}

print(f"Processing data for {target_year}...")
for date_idx, row in df.iterrows():

    if date_idx.year != target_year:
        continue
        
    m = date_idx.month
    
    sst = row.get('sst')
    chlor = row.get('chlorophyll')
    ssl = row.get('ssl')

    # Basic validity checks
    if sst is None or chlor is None or ssl is None: continue
    if isinstance(sst, float) and np.isnan(sst): continue
    if isinstance(chlor, float) and np.isnan(chlor): continue
    if isinstance(ssl, float) and np.isnan(ssl): continue
    
    sst_arr = np.array(sst, dtype=float)
    chlor_arr = np.array(chlor, dtype=float)
    ssl_arr = np.array(ssl, dtype=float)
    
    # Process SSL (blur) as in the reference analysis to fill small gaps
    ssl_nan_mask = np.isnan(ssl_arr)
    if not ssl_nan_mask.all():
        ssl_filled = ssl_arr.copy()
        ssl_filled[ssl_nan_mask] = np.nanmean(ssl_arr)
        ssl_blurred = gaussian_filter(ssl_filled, sigma=2.0)
        ssl_blurred[ssl_nan_mask] = np.nan
        ssl_arr = ssl_blurred

    month_data[m]['sst'].append(sst_arr)
    month_data[m]['chlor'].append(chlor_arr)
    month_data[m]['ssl'].append(ssl_arr)

# Compute aggregated means
# This will now reflect only the selected year
monthly_means = {m: {} for m in months}
vars_to_aggregate = ['sst', 'chlor', 'ssl']

for m in months:
    for var in vars_to_aggregate:
        grids = month_data[m][var]
        if len(grids) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                # Stack and Mean
                stack = np.stack(grids, axis=0)
                mean_grid = np.nanmean(stack, axis=0)
                monthly_means[m][var] = mean_grid
        else:
            monthly_means[m][var] = None

# Pre-calculate log10(chlor) for visualization
for m in months:
    if monthly_means[m]['chlor'] is not None:
        monthly_means[m]['log_chlor'] = np.log10(monthly_means[m]['chlor'])
    else:
        monthly_means[m]['log_chlor'] = None

print(f"Monthly means for {target_year} computed.")

def plot_monthly_variable_raw(var_key, title_str, unit_label, cmap, vmin=None, vmax=None):
    month_names_full = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
    
    lat_min, lon_min = 19.57871, -92.20216
    lat_max, lon_max = 22.86906, -85.61256
    
    if vmin is None or vmax is None:
        all_vals = []
        for m in range(1, 13):
            grid = monthly_means[m].get(var_key)
            if grid is not None:
                all_vals.append(grid.flatten())
        if all_vals:
            all_vals = np.concatenate(all_vals)
            if vmin is None: vmin = np.nanpercentile(all_vals, 1)
            if vmax is None: vmax = np.nanpercentile(all_vals, 99)
            
    print(f"Plotting {title_str}: vmin={vmin:.2f}, vmax={vmax:.2f}")

    fig, axes = plt.subplots(3, 4, figsize=(28, 10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=400)
    
    mesh_for_cbar = None

    for month_num, ax in enumerate(axes.ravel(), start=1):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Allow axes to stretch vertically/horizontally rather than enforcing equal-aspect
        try:
            ax.set_aspect('auto')
        except Exception:
            pass

        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
        ax.gridlines(draw_labels=False, linewidth=0.45, color='white', alpha=0.55, linestyle='--', zorder=2)
        
        data_grid = monthly_means[month_num].get(var_key)
        
        if data_grid is None:
            ax.set_title(f"{month_names_full[month_num-1]}\nNo Data", fontsize=10)
            continue
            
        mesh = ax.pcolormesh(
            lon_grid_2d, lat_grid_2d, data_grid,
            cmap=cmap, transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, shading='auto', zorder=1
        )
        
        if mesh_for_cbar is None:
            mesh_for_cbar = mesh
            
        ax.set_title(f"{month_names_full[month_num-1]}", fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.9, wspace=0.06, hspace=0.1)
    if mesh_for_cbar is not None:
        cax = fig.add_axes([0.915, 0.04, 0.012, 0.88])
        cbar = fig.colorbar(mesh_for_cbar, cax=cax)
        cbar.set_label(unit_label, fontsize=12)

    # Include the selected year in the suptitle (target_year is defined earlier)
    # fig.suptitle(f'Monthly Mean {title_str} (Jan–Dec) ({target_year})', fontsize=16, y=0.98)

    # plt.show()

    plt.savefig(f"./yucatan/experiments/visualization/raw/figs/{var_key}")

plot_monthly_variable_raw('log_chlor', 'Chlorophyll-a (Log Scale)', 'log10(mg/m³)', 'viridis')
plot_monthly_variable_raw('sst', 'Sea Surface Temperature', '°C', 'RdYlBu_r')
plot_monthly_variable_raw('ssl', 'Sea Surface Level (SSL)', 'm', 'magma')