"""
utils.py
--------
Shared helpers for loading optical / SAR field-crop patches and their labels.

EDIT ME: the exact folder layout and label format depend on how the
YieldSAT-derived dataset was handed to you. Adjust `load_patch`,
`load_label_table`, etc. to match. The rest of the pipeline
(optical_features.py, sar_features.py, fusion.py, train_classifier.py)
only depends on the *return types* documented below, so once these loaders
are correct everything downstream should work unchanged.
"""

import os
import glob
import numpy as np
import pandas as pd

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


def load_patch(path):
    """
    Load a single-field raster patch (optical or SAR) as a numpy array.

    Returns
    -------
    arr : np.ndarray, shape (bands, H, W)
    """
    if rasterio is None:
        raise ImportError("pip install rasterio --break-system-packages")
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    return arr.astype(np.float32)


def list_field_patches(root_dir, ext="tif"):
    """
    Return a sorted list of patch file paths under root_dir.
    EDIT: adjust the glob pattern to match your actual file naming,
    e.g. '<field_id>_<date>_optical.tif'.
    """
    return sorted(glob.glob(os.path.join(root_dir, f"*.{ext}")))


def field_id_from_path(path):
    """
    EDIT ME: parse the field/patch identifier out of a filename so that
    optical and SAR patches for the *same* field can be matched up.
    """
    fname = os.path.basename(path)
    return os.path.splitext(fname)[0]


def load_label_table(csv_path):
    """
    Load the crop-type / yield label table.

    Expected columns (EDIT to match the real file):
        field_id, crop_type, yield_value (optional)

    Returns
    -------
    pd.DataFrame indexed by field_id
    """
    df = pd.read_csv(csv_path)
    df = df.set_index("field_id")
    return df


def yield_to_class(yield_mask, n_bins=3):
    """
    Optional helper: if you need to derive a classification label from a
    per-pixel yield mask (rather than using a provided crop-type label),
    bin the mean yield of the patch into n_bins classes (e.g. low/med/high).
    """
    mean_yield = np.nanmean(yield_mask)
    # EDIT: replace with bin edges computed from your actual yield distribution
    edges = np.linspace(np.nanmin(yield_mask), np.nanmax(yield_mask), n_bins + 1)
    label = np.digitize(mean_yield, edges[1:-1])
    return int(label)
