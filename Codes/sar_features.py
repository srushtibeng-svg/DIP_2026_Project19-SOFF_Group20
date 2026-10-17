"""
sar_features.py

Feature extraction from Sentinel-1 VV and VH raster patches.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_band_statistics(band, prefix):
    """Calculate basic statistics for one SAR band."""

    band = np.asarray(band, dtype=float)
    valid = band[np.isfinite(band)]

    if valid.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan
        }

    return {
        f"{prefix}_mean": float(valid.mean()),
        f"{prefix}_std": float(valid.std()),
        f"{prefix}_min": float(valid.min()),
        f"{prefix}_max": float(valid.max())
    }


def extract_ratio_features(vv, vh):
    """Calculate VV/VH ratio using linear backscatter values."""

    vv = np.asarray(vv, dtype=float)
    vh = np.asarray(vh, dtype=float)

    valid = (
        np.isfinite(vv)
        & np.isfinite(vh)
        & (np.abs(vh) > 1e-8)
    )

    if not valid.any():
        return {
            "vv_vh_ratio_mean": np.nan,
            "vv_vh_ratio_std": np.nan
        }

    ratio = vv[valid] / vh[valid]

    return {
        "vv_vh_ratio_mean": float(ratio.mean()),
        "vv_vh_ratio_std": float(ratio.std())
    }


def extract_glcm_features(
    band,
    prefix,
    levels=32
):
    """Extract texture features from a 2D SAR band."""

    band = np.asarray(band, dtype=float)

    if band.ndim != 2:
        raise ValueError(
            "Expected a 2D image band."
        )

    finite = band[np.isfinite(band)]

    props = [
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]

    if finite.size == 0:
        return {
            f"{prefix}_glcm_{p}": np.nan
            for p in props
        }

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        quantized = np.zeros(
            band.shape,
            dtype=np.uint8
        )
    else:
        scaled = np.clip(
            (band - low) / (high - low),
            0,
            1
        )

        quantized = np.zeros(
            band.shape,
            dtype=np.uint8
        )

        valid = np.isfinite(band)

        quantized[valid] = (
            scaled[valid] * (levels - 1)
        ).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=levels,
        symmetric=True,
        normed=True
    )

    features = {}

    for prop in props:
        features[f"{prefix}_glcm_{prop}"] = float(
            np.mean(graycoprops(glcm, prop))
        )

    return features


def extract_sar_features(patch):
    """
    Extract features from a SAR patch.

    Expected shape:
        (2, height, width)

    Band order:
        0 = VV
        1 = VH
    """

    patch = np.asarray(patch, dtype=float)

    if patch.ndim != 3 or patch.shape[0] < 2:
        raise ValueError(
            "SAR patch must contain VV and VH bands."
        )

    vv = patch[0]
    vh = patch[1]

    features = {}

    features.update(
        extract_band_statistics(vv, "vv")
    )

    features.update(
        extract_band_statistics(vh, "vh")
    )

    features.update(
        extract_ratio_features(vv, vh)
    )

    features.update(
        extract_glcm_features(vv, "vv")
    )

    features.update(
        extract_glcm_features(vh, "vh")
    )

    return features
