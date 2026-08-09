
# %% imports
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

import joblib

PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v3.pkl"
MODEL_PATH = r"yucatan\experiments\isolation_forest\iforest_pipeline2.joblib"

set_config(display="text")

# %% load pickle
with open(PATH, "rb") as f:
    blob = pickle.load(f)

df = blob["data"]
metadata = blob["metadata"]
df = df[(df.index.year >= 2010) & (df.index.year <= 2019)]
print(f"Rows after date trim: {len(df):,}")

grid_shape = np.asarray(df["sst"].iloc[0]).shape  # SST is L4, well processed -> used as reference shape

# %% drop rows with invalid data
def shape_ok(a):
    return a is not None and np.asarray(a).shape == grid_shape

keep = df["sst"].apply(shape_ok) & df["chlorophyll"].apply(shape_ok) & df["ssl"].apply(shape_ok)
df = df[keep]

# %% flatten
sst_stack = np.stack(df["sst"].to_numpy()).astype(float)
chlor_stack = np.stack(df["chlorophyll"].to_numpy()).astype(float)
ssl_stack = np.stack(df["ssl"].to_numpy()).astype(float)

valid = (
    ~np.isnan(sst_stack) & ~np.isnan(chlor_stack) & ~np.isnan(ssl_stack)
    & (chlor_stack > 0)
)

t_idx, row_idx, col_idx = np.where(valid)

doy = pd.DatetimeIndex(df.index).dayofyear.to_numpy()
doy_per_row = doy[t_idx]

pixels = pd.DataFrame({
    "date": np.asarray(df.index)[t_idx],
    "row": row_idx,
    "col": col_idx,
    "sst": sst_stack[valid],
    "chlor": chlor_stack[valid],
    "ssl": ssl_stack[valid],
})

print(f"Total valid pixel-observations: {len(pixels):,}")


# %% deseasonalizer: per-pixel, circular 7-day-window, MEDIAN climatology
class ClimatologyDeseasonalizer(BaseEstimator, TransformerMixin):
    def __init__(self, value_cols, window=7):
        self.value_cols = value_cols
        self.window = window

    def fit(self, X, y=None):
        doy = np.clip(pd.DatetimeIndex(X["date"]).dayofyear.to_numpy(), None, 365) - 1
        X = X.assign(doy=doy)

        # sparse per-(row, col, doy) median first -- cheap via groupby
        daily = X.groupby(["row", "col", "doy"])[self.value_cols].median().reset_index()

        self.n_rows_ = int(X["row"].max()) + 1
        self.n_cols_ = int(X["col"].max()) + 1
        self.climatology_ = {}

        for col in self.value_cols:
            grid = np.full((self.n_rows_, self.n_cols_, 365), np.nan)
            grid[daily["row"], daily["col"], daily["doy"]] = daily[col]
            self.climatology_[col] = self._circular_rolling_median(grid)

        return self

    def _circular_rolling_median(self, grid):
        half = self.window // 2
        padded = np.concatenate([grid[..., -half:], grid, grid[..., :half]], axis=-1)
        out = np.full_like(grid, np.nan)
        for i in range(grid.shape[-1]):
            out[..., i] = np.nanmedian(padded[..., i:i + self.window], axis=-1)
        return out

    def transform(self, X):
        doy = np.clip(pd.DatetimeIndex(X["date"]).dayofyear.to_numpy(), None, 365) - 1
        row = X["row"].to_numpy()
        col = X["col"].to_numpy()
        out = X.copy()
        for c in self.value_cols:
            clim_vals = self.climatology_[c][row, col, doy]
            out[f"{c}_anom"] = X[c].to_numpy() - clim_vals
        return out


# %% fit deseasonalizer on FULL pixel set (needs full per-pixel history,
#    NOT the IF fit subsample -- otherwise climatology is noisy/sparse)
deseasonalizer = ClimatologyDeseasonalizer(value_cols=["sst", "ssl"], window=7)
deseasonalizer.fit(pixels)
pixels_deseasoned = deseasonalizer.transform(pixels)

# %% drop rows with no climatology coverage for their (row, col, doy)
n_before = len(pixels_deseasoned)
pixels_deseasoned = pixels_deseasoned.dropna(subset=["sst_anom", "ssl_anom"])
n_after = len(pixels_deseasoned)
dropped_pct = 100 * (n_before - n_after) / n_before
print(f"Dropped {n_before - n_after:,} rows ({dropped_pct:.2f}%) with no climatology coverage")
if dropped_pct > 2:
    print("WARNING: coverage gap is larger than expected -- check grid coverage / date range")

# %% preprocessing and model pipeline
# time_sin/time_cos removed: seasonality is now handled by deseasonalization,
# so keeping them risked the model re-learning season as a separate signal.
# chlor kept in for now per current decision (TODO: ablation with/without).
skewed_features = ["chlor", "ssl_anom"]
normal_features = ["sst_anom"]
features = skewed_features + normal_features

preprocessor = ColumnTransformer([
    ("skewed", PowerTransformer(method="yeo-johnson", standardize=True), skewed_features),
    ("normal", StandardScaler(), normal_features),
])
preprocessor.set_output(transform="pandas")

iforest_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("iforest", IsolationForest(
        n_estimators=200,
        contamination="auto",   # don't rely on .predict(); threshold decision_function() yourself downstream
        random_state=42,
        n_jobs=-1,
    )),
])

print(iforest_pipeline)

# %% fit IF on a subsample of the DESEASONALIZED data
SAMPLE_SIZE = 20_000_000
pixels_fit = pixels_deseasoned.sample(n=min(SAMPLE_SIZE, len(pixels_deseasoned)), random_state=42)
X_fit = pixels_fit[features]

iforest_pipeline.fit(X_fit)

# %% save BOTH the deseasonalizer and the IF pipeline
# both are required at inference: deseasonalizer to compute sst_anom/ssl_anom
# for new data, then iforest_pipeline to score those anomalies.
full_bundle = {
    "deseasonalizer": deseasonalizer,
    "iforest_pipeline": iforest_pipeline,
    "features": features,
}
joblib.dump(full_bundle, MODEL_PATH)
print(f"Saved bundle (deseasonalizer + iforest_pipeline) to {MODEL_PATH}")

# %% --- inference example (for reference) ---
# bundle = joblib.load(MODEL_PATH)
# new_df = bundle["deseasonalizer"].transform(new_df)          # adds sst_anom, ssl_anom
# new_df = new_df.dropna(subset=["sst_anom", "ssl_anom"])      # handle coverage gaps
# scores = bundle["iforest_pipeline"].decision_function(new_df[bundle["features"]])
# anomaly_score = -scores  # flip so higher = more anomalous