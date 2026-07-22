"""Shared pieces between train_model.py and predict.py.

GeoCluster must live in its own importable module (not inside a __main__
script) so joblib can unpickle it later from a different entry point.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

RANDOM_STATE = 42
N_GEO_CLUSTERS = 8
SMOOTHING_ALPHA = 10  # Laplace smoothing strength for county historical rate


class GeoCluster(BaseEstimator, TransformerMixin):
    """Assigns each (lat, lon) to a KMeans region cluster, fit on training data only."""

    def __init__(self, n_clusters=N_GEO_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X)
        return self

    def transform(self, X):
        labels = self.kmeans_.predict(np.asarray(X))
        return labels.reshape(-1, 1).astype(str)


# Generous CA-plus-border-states bounding box; anything outside this after
# the fixes below is treated as unrecoverable bad data, not a real location.
CA_LAT_RANGE = (30.0, 44.0)
CA_LON_RANGE = (-126.0, -112.0)


def clean_lat_lon(df, lat_col="Latitude", lon_col="Longitude"):
    """Fix known data-quality issues in the CAL FIRE lat/lon columns:
    - (0, 0) "null island" rows are missing data, not a real location
    - some rows have latitude/longitude swapped (lat > 90 is impossible)
    - a few swapped rows also lost the negative sign on longitude
    Anything that still falls outside a generous CA bounding box afterwards
    is set to NaN so downstream code drops or imputes it instead of silently
    treating garbage coordinates as a real fire location."""
    df = df.copy()
    swapped = (df[lat_col].abs() > 90) & (df[lon_col].abs() <= 90)
    df.loc[swapped, [lat_col, lon_col]] = df.loc[swapped, [lon_col, lat_col]].values

    flipped_sign = (df[lon_col] > 0) & (df[lon_col].between(-CA_LON_RANGE[1], -CA_LON_RANGE[0]))
    df.loc[flipped_sign, lon_col] = -df.loc[flipped_sign, lon_col]

    null_island = (df[lat_col] == 0) & (df[lon_col] == 0)
    out_of_range = (
        ~df[lat_col].between(*CA_LAT_RANGE) | ~df[lon_col].between(*CA_LON_RANGE)
    )
    df.loc[null_island | out_of_range, [lat_col, lon_col]] = np.nan
    return df


def add_county_history_rate(df, county_col, date_col, target_col, alpha=SMOOTHING_ALPHA):
    """Smoothed rate of `target_col` in `county_col`, using only rows strictly
    earlier in `date_col`. Safe to compute on the whole dataset because it
    only ever looks backward in time relative to each row."""
    df = df.sort_values(date_col).reset_index(drop=True)
    grp = df.groupby(county_col)[target_col]
    prior_sum = grp.cumsum() - df[target_col]
    prior_count = grp.cumcount()
    global_rate = df[target_col].expanding().mean().shift(1).fillna(df[target_col].mean())
    df["CountyHistoricalRate"] = (prior_sum + alpha * global_rate) / (prior_count + alpha)
    return df
