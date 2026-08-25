# %% imports
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

import joblib

PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v4.pkl"
MODEL_PATH = r"F:\Programming\GitHub\nasa-murep\yucatan\experiments\isolation_forest\models\iforest_pipeline5.joblib"

set_config(display="text")

# %% load pickle
with open(PATH, "rb") as f:
    blob = pickle.load(f)

df = blob["data"]
metadata = blob["metadata"]
df = df[(df.index.year >= 2010) & (df.index.year <= 2019)]
print(f"Rows after date trim: {len(df):,}")

grid_shape = np.asarray(df["sst"].iloc[0]).shape  # SST is L4 data (reliable) so its used as the reference shape

# %% drop rows with invalid data
def shape_ok(a):
    return a is not None and np.asarray(a).shape == grid_shape

keep = df["sst"].apply(shape_ok) & df["chlorophyll"].apply(shape_ok) & df["ssl"].apply(shape_ok)
df = df[keep]

# %% stack into 3D arrays (time, row, col)
sst_stack = np.stack(df["sst"].to_numpy()).astype(float)
chlor_stack = np.stack(df["chlorophyll"].to_numpy()).astype(float)
ssl_stack = np.stack(df["ssl"].to_numpy()).astype(float)

# %% same-day, whole-domain anomaly
def same_day_anomaly(stack):
    daily_median = np.nanmedian(stack, axis=(1, 2), keepdims=True)
    return stack - daily_median

sst_anom_stack = same_day_anomaly(sst_stack)
ssl_anom_stack = same_day_anomaly(ssl_stack)

# %% flatten to per-pixel-observation rows
valid = (
    ~np.isnan(sst_anom_stack) & ~np.isnan(ssl_anom_stack) & ~np.isnan(chlor_stack)
    & (chlor_stack > 0)
)
t_idx, row_idx, col_idx = np.where(valid)

pixels = pd.DataFrame({
    "date": np.asarray(df.index)[t_idx],
    "row": row_idx,
    "col": col_idx,
    "sst_anom": sst_anom_stack[valid],
    "ssl_anom": ssl_anom_stack[valid],
    "chlor": chlor_stack[valid], # descriptive/val only
})

print(f"Total valid pixel-observations: {len(pixels):,}")

# %% preprocessing and model pipeline
features = ["sst_anom", "ssl_anom"]

preprocessor = ColumnTransformer([
    ("anom", PowerTransformer(method="yeo-johnson", standardize=True), features),
])
preprocessor.set_output(transform="pandas")

iforest_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("iforest", IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )),
])

print(iforest_pipeline)

# %% fit on a subsample
SAMPLE_SIZE = 20_000_000
pixels_fit = pixels.sample(n=min(SAMPLE_SIZE, len(pixels)), random_state=42)
X_fit = pixels_fit[features]

iforest_pipeline.fit(X_fit)

# %% save
joblib.dump({"iforest_pipeline": iforest_pipeline, "features": features}, MODEL_PATH)
print(f"Saved pipeline to {MODEL_PATH}")
