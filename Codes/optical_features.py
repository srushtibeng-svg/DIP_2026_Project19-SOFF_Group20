"""
optical_features.py
--------------------
Classical hand-crafted feature extraction from Sentinel-2 optical patches.

Assumes a Sentinel-2 patch with at least these bands (EDIT band indices to
match your actual stack order):
    B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)  -- minimum for NDVI/NDWI
Add B5/B8A/B11/B12 etc. if your provided stack includes them.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

# EDIT: set these to the actual band order in your provided GeoTIFFs
BAND_IDX = {"blue": 0, "green": 1, "red": 2, "nir": 3}


def _safe_ratio(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


def spectral_indices(patch):
    """
    patch: np.ndarray (bands, H, W)
    Returns dict of scalar indices (mean over the patch).
    """
    blue = patch[BAND_IDX["blue"]]
    green = patch[BAND_IDX["green"]]
    red = patch[BAND_IDX["red"]]
    nir = patch[BAND_IDX["nir"]]

    ndvi = _safe_ratio(nir, red)
    ndwi = _safe_ratio(green, nir)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-6)
    savi = _safe_ratio(nir, red) * 1.5  # simplified SAVI, L=0.5 folded in loosely; refine if needed

    return {
        "ndvi_mean": float(np.nanmean(ndvi)),
        "ndvi_std": float(np.nanstd(ndvi)),
        "ndwi_mean": float(np.nanmean(ndwi)),
        "evi_mean": float(np.nanmean(evi)),
        "savi_mean": float(np.nanmean(savi)),
    }


def band_statistics(patch):
    """Basic per-band mean/std — cheap but often useful baseline features."""
    feats = {}
    for name, idx in BAND_IDX.items():
        band = patch[idx]
        feats[f"{name}_mean"] = float(np.nanmean(band))
        feats[f"{name}_std"] = float(np.nanstd(band))
    return feats


def glcm_texture(patch, band="nir", levels=32, distances=(1,), angles=(0,)):
    """
    Gray-Level Co-occurrence Matrix texture features on one band.
    Quantizes the band to `levels` gray levels before computing GLCM.
    """
    band_arr = patch[BAND_IDX[band]]
    finite = band_arr[np.isfinite(band_arr)]
    if finite.size == 0:
        return {f"glcm_{band}_{p}": 0.0 for p in
                ["contrast", "homogeneity", "energy", "correlation"]}

    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
    scaled = np.clip((band_arr - lo) / (hi - lo + 1e-6), 0, 1)
    quantized = (scaled * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(quantized, distances=list(distances), angles=list(angles),
                         levels=levels, symmetric=True, normed=True)

    feats = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{band}_{prop}"] = float(np.mean(graycoprops(glcm, prop)))
    return feats


def extract_optical_features(patch):
    """
    Main entry point: given one optical patch (bands, H, W), return a flat
    dict of feature_name -> value. Combine into a single feature vector
    downstream (see fusion.py).
    """
    feats = {}
    feats.update(band_statistics(patch))
    feats.update(spectral_indices(patch))
    feats.update(glcm_texture(patch, band="nir"))
    feats.update(glcm_texture(patch, band="red"))
    return feats
