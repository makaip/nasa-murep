import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ClimatologyDeseasonalizer(BaseEstimator, TransformerMixin):
    def __init__(self, value_cols, window=7):  # weekly rolling average
        self.value_cols = value_cols
        self.window = window

    def fit(self, X, y=None):
        doy = np.clip(pd.DatetimeIndex(X["date"]).dayofyear.to_numpy(), None, 365) - 1
        X = X.assign(doy=doy)
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