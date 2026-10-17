"""
utils.py

Shared utilities for loading YieldSAT CSV datasets
and Sentinel-1 / Sentinel-2 raster patches.
"""

import os
import glob
import numpy as np
import pandas as pd

try:
    import rasterio
except ImportError:
    rasterio = None


def load_dataset(csv_path):
    """Load a CSV dataset and validate that it is not empty."""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def load_label_table(csv_path):
    """
    Load a dataset containing crop labels.

    The actual field identifier is retained in the columns.
    """

    df = load_dataset(csv_path)

    required_columns = ["crop"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found."
            )

    return df


def load_patch(path):
    """
    Load a raster patch.

    Returns:
        NumPy array with shape (bands, height, width).
    """

    if rasterio is None:
        raise ImportError(
            "Install rasterio before loading raster patches."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        patch = src.read().astype(np.float32)

    return patch


def list_field_patches(root_dir, ext="tif"):
    """List raster patches in a directory."""

    pattern = os.path.join(root_dir, f"*.{ext}")

    return sorted(glob.glob(pattern))


def field_id_from_path(path):
    """Return the filename without its extension."""

    return os.path.splitext(
        os.path.basename(path)
    )[0]


def validate_patch(patch):
    """Check whether a raster patch has a valid shape."""

    patch = np.asarray(patch)

    if patch.ndim != 3:
        raise ValueError(
            "Expected patch shape (bands, height, width)."
        )

    if patch.shape[0] == 0:
        raise ValueError("Patch contains no bands.")

    return patch


def get_crop_labels(df):
    """Return crop labels from the dataset."""

    if "crop" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'crop' column."
        )

    return df["crop"].astype(str)


def print_dataset_summary(df, dataset_name="Dataset"):
    """Display basic dataset information."""

    print(f"\n--- {dataset_name} Summary ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    if "crop" in df.columns:
        print("\nCrop distribution:")
        print(df["crop"].value_counts())


def yield_to_class(yield_mask, n_bins=3):
    """
    Optional utility to convert a numerical yield array
    into low, medium and high classes.

    This function is not required for crop-type classification.
    """

    values = np.asarray(yield_mask, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No valid yield values found.")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    edges = np.linspace(
        values.min(),
        values.max(),
        n_bins + 1
    )

    label = np.digitize(
        values.mean(),
        edges[1:-1]
    )

    return int(label)
