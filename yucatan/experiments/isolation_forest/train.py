
# %% imports
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

import joblib

PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v3.pkl"
MODEL_PATH = r".\yucatan\experiments\isolation_forest\iforest_pipeline1.joblib"

set_config(display="text")

# %% load pickle
with open(PATH, "rb") as f:
    blob = pickle.load(f)

df = blob["data"]
metadata = blob["metadata"]
df = df[(df.index.year >= 2010) & (df.index.year <= 2019)]
print(f"Rows after date trim: {len(df):,}")

grid_shape = np.asarray(df["sst"].iloc[0]).shape  # using SST as the shape because its L4 and very well processed

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
    "time_sin": np.sin(2 * np.pi * doy_per_row / 365.25),
    "time_cos": np.cos(2 * np.pi * doy_per_row / 365.25),
})

# %% preprocessing and model pipeline
skewed_features = ["chlor", "ssl"]
normal_features = ["sst", "time_sin", "time_cos"]
features = skewed_features + normal_features

preprocessor = ColumnTransformer([
    ("skewed", PowerTransformer(method="yeo-johnson", standardize=True), skewed_features),
    ("normal", StandardScaler(), normal_features),
])
preprocessor.set_output(transform="pandas")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("iforest", IsolationForest(
        n_estimators=200,
        contamination='auto',
        random_state=42,
        n_jobs=-1,
    )),
])

print(pipeline)

# %% fit to subsample
SAMPLE_SIZE = 20_000_000
pixels_fit = pixels.sample(n=min(SAMPLE_SIZE, len(pixels)), random_state=42)
X_fit = pixels_fit[features]

pipeline.fit(X_fit)

# %% save the fitted pipeline (preprocessor + iforest bundled together)
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved pipeline to {MODEL_PATH}")
