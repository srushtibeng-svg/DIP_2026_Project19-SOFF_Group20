"""
fusion.py
---------
Combine optical and SAR feature dicts/vectors into fused feature vectors.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_feature_dataframe(records):
    """
    records: list of dicts, each with keys:
        field_id, label, optical_feats (dict), sar_feats (dict)

    Returns a wide DataFrame with columns:
        field_id, label, opt_<name>..., sar_<name>...
    """
    rows = []
    for r in records:
        row = {"field_id": r["field_id"], "label": r["label"]}
        row.update({f"opt_{k}": v for k, v in r["optical_feats"].items()})
        row.update({f"sar_{k}": v for k, v in r["sar_feats"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def split_modalities(df):
    """
    Given the wide DataFrame from build_feature_dataframe, return
    (X_optical, X_sar, X_fused, y) as numpy arrays / pandas objects.
    """
    opt_cols = [c for c in df.columns if c.startswith("opt_")]
    sar_cols = [c for c in df.columns if c.startswith("sar_")]

    X_optical = df[opt_cols].values
    X_sar = df[sar_cols].values
    X_fused = df[opt_cols + sar_cols].values
    y = df["label"].values
    return X_optical, X_sar, X_fused, y, opt_cols, sar_cols


def scale_features(X_train, X_test):
    """Standardize features (fit on train, apply to test) — do this per split
    to avoid leakage."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def reduce_with_pca(X_train, X_test, n_components=0.95):
    """
    Optional dimensionality reduction before/after fusion, useful once the
    fused vector gets long. n_components can be an int or a float
    (fraction of variance to keep, e.g. 0.95).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)
    return X_train_p, X_test_p, pca
