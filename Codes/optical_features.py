"""
optical_features.py

Extract spectral indices, band statistics and GLCM
texture features from Sentinel-2 raster patches.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def safe_ratio(a, b, eps=1e-6):
    """Calculate (a - b) / (a + b) safely."""

    denominator = a + b

    result = np.full(
        np.broadcast_shapes(a.shape, b.shape),
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(denominator) > eps)
    )

    result[valid] = (
        (a[valid] - b[valid])
        / denominator[valid]
    )

    return result


def get_band(patch, band_index):
    """Return one band from a raster patch."""

    patch = np.asarray(patch, dtype=float)

    if patch.ndim != 3:
        raise ValueError(
            "Patch must have shape (bands, height, width)."
        )

    if band_index < 0 or band_index >= patch.shape[0]:
        raise ValueError(
            f"Invalid band index: {band_index}"
        )

    return patch[band_index]


def spectral_indices(patch, band_idx):
    """
    Extract mean and standard deviation of spectral indices.

    band_idx example:
        {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3
        }

    The indices must match the actual GeoTIFF band order.
    """

    blue = get_band(patch, band_idx["blue"])
    green = get_band(patch, band_idx["green"])
    red = get_band(patch, band_idx["red"])
    nir = get_band(patch, band_idx["nir"])

    ndvi = safe_ratio(nir, red)
    ndwi = safe_ratio(green, nir)

    evi_denominator = (
        nir + 6 * red - 7.5 * blue + 1
    )

    evi = np.full_like(nir, np.nan, dtype=float)

    valid_evi = (
        np.isfinite(nir)
        & np.isfinite(red)
        & np.isfinite(blue)
        & (np.abs(evi_denominator) > 1e-6)
    )

    evi[valid_evi] = (
        2.5
        * (nir[valid_evi] - red[valid_evi])
        / evi_denominator[valid_evi]
    )

    # Standard SAVI formula with L = 0.5.
    savi_denominator = nir + red + 0.5

    savi = np.full_like(nir, np.nan, dtype=float)

    valid_savi = (
        np.isfinite(nir)
        & np.isfinite(red)
        & (np.abs(savi_denominator) > 1e-6)
    )

    savi[valid_savi] = (
        1.5
        * (nir[valid_savi] - red[valid_savi])
        / savi_denominator[valid_savi]
    )

    indices = {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "evi": evi,
        "savi": savi
    }

    features = {}

    for name, values in indices.items():
        features[f"{name}_mean"] = (
            float(np.nanmean(values))
            if np.isfinite(values).any()
            else np.nan
        )

        features[f"{name}_std"] = (
            float(np.nanstd(values))
            if np.isfinite(values).any()
            else np.nan
        )

    return features


def band_statistics(patch, band_idx):
    """Calculate mean and standard deviation for each selected band."""

    features = {}

    for name, index in band_idx.items():
        band = get_band(patch, index)
        valid = band[np.isfinite(band)]

        if valid.size == 0:
            features[f"{name}_mean"] = np.nan
            features[f"{name}_std"] = np.nan
        else:
            features[f"{name}_mean"] = float(valid.mean())
            features[f"{name}_std"] = float(valid.std())

    return features


def glcm_texture(
    patch,
    band_idx,
    band="nir",
    levels=32
):
    """Extract GLCM texture features from one band."""

    if band not in band_idx:
        raise ValueError(
            f"Band '{band}' is not present in band_idx."
        )

    band_array = get_band(
        patch,
        band_idx[band]
    )

    finite = band_array[np.isfinite(band_array)]

    properties = [
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]

    if finite.size == 0:
        return {
            f"glcm_{band}_{prop}": np.nan
            for prop in properties
        }

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        quantized = np.zeros(
            band_array.shape,
            dtype=np.uint8
        )
    else:
        scaled = np.clip(
            (band_array - low) / (high - low),
            0,
            1
        )

        quantized = np.zeros(
            band_array.shape,
            dtype=np.uint8
        )

        valid = np.isfinite(band_array)

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

    for prop in properties:
        features[f"glcm_{band}_{prop}"] = float(
            np.mean(graycoprops(glcm, prop))
        )

    return features


def extract_optical_features(
    patch,
    band_idx
):
    """Return a dictionary of optical features."""

    features = {}

    features.update(
        band_statistics(patch, band_idx)
    )

    features.update(
        spectral_indices(patch, band_idx)
    )

    for band in ["nir", "red"]:
        if band in band_idx:
            features.update(
                glcm_texture(
                    patch,
                    band_idx,
                    band=band
                )
            )

    return features
