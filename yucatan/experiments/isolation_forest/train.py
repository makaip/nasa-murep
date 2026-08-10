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

from deseasonalizer import ClimatologyDeseasonalizer

PATH = r"E:\satdata\Custom\yucatan_sst_chlor_cdomslope_ssl_bathy_2010-2019_v3.pkl"
MODEL_PATH = r".\yucatan\experiments\isolation_forest\models\iforest_pipeline4.joblib"

set_config(display="text")

# %% load pickle
with open(PATH, "rb") as f:
    blob = pickle.load(f)

df = blob["data"]
metadata = blob["metadata"]
df = df[(df.index.year >= 2010) & (df.index.year <= 2019)]
print(f"Rows after date trim: {len(df):,}")

grid_shape = np.asarray(df["sst"].iloc[0]).shape  # SST is L4, well processed. used as reference shape

# %% drop rows with invalid data
def shape_ok(a):
    return a is not None and np.asarray(a).shape == grid_shape

keep = df["sst"].apply(shape_ok) & df["ssl"].apply(shape_ok)
df = df[keep]

# %% flatten
sst_stack = np.stack(df["sst"].to_numpy()).astype(float)
ssl_stack = np.stack(df["ssl"].to_numpy()).astype(float)

valid = ~np.isnan(sst_stack) & ~np.isnan(ssl_stack)

t_idx, row_idx, col_idx = np.where(valid)

doy = pd.DatetimeIndex(df.index).dayofyear.to_numpy()
doy_per_row = doy[t_idx]

pixels = pd.DataFrame({
    "date": np.asarray(df.index)[t_idx],
    "row": row_idx,
    "col": col_idx,
    "sst": sst_stack[valid],
    "ssl": ssl_stack[valid],
})

print(f"Total valid pixel-observations: {len(pixels):,}")

# %% fit deseasonalizer on full pixel set
deseasonalizer = ClimatologyDeseasonalizer(value_cols=["sst", "ssl"], window=7)
deseasonalizer.fit(pixels)
pixels_deseasoned = deseasonalizer.transform(pixels)

# %% drop rows with no climatology coverage for their (row, col, doy)
n_before = len(pixels_deseasoned)
pixels_deseasoned = pixels_deseasoned.dropna(subset=["sst_anom", "ssl_anom"])
n_after = len(pixels_deseasoned)
dropped_pct = 100 * (n_before - n_after) / n_before
print(f"Dropped {n_before - n_after:,} rows ({dropped_pct:.2f}%) with no climatology coverage") # this is always 0 anyways, but just in case

# %% preprocessing and model pipeline
skewed_features = ["ssl_anom"]
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
        contamination="auto",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )),
])

print(iforest_pipeline)

# %% fit IF on a subsample of the deseasonalized data
SAMPLE_SIZE = 20_000_000
pixels_fit = pixels_deseasoned.sample(n=min(SAMPLE_SIZE, len(pixels_deseasoned)), random_state=42)
X_fit = pixels_fit[features]

iforest_pipeline.fit(X_fit)

# %% save deseasonalizer and IF pipeline
full_bundle = {
    "deseasonalizer": deseasonalizer,
    "iforest_pipeline": iforest_pipeline,
    "features": features,
}
joblib.dump(full_bundle, MODEL_PATH)
print(f"Saved bundle to {MODEL_PATH}")
