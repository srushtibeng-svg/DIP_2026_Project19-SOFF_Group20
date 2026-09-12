"""
sar_features.py
----------------
Classical hand-crafted feature extraction from Sentinel-1 SAR patches
(exported via Google Earth Engine — see gee_sar_export.py).

Assumes a two-band stack: [VV, VH] backscatter (linear or dB — EDIT
`IS_DB` below to match whatever units you exported from GEE).
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

SAR_BAND_IDX = {"vv": 0, "vh": 1}
IS_DB = True  # set False if you exported linear backscatter instead of dB


def to_db(x):
    """Convert linear backscatter to dB if needed."""
    return 10 * np.log10(np.clip(x, 1e-6, None))


def backscatter_statistics(patch):
    """Mean/std/median per polarization + VV/VH ratio."""
    vv = patch[SAR_BAND_IDX["vv"]]
    vh = patch[SAR_BAND_IDX["vh"]]

    if not IS_DB:
        vv, vh = to_db(vv), to_db(vh)

    feats = {
        "vv_mean": float(np.nanmean(vv)),
        "vv_std": float(np.nanstd(vv)),
        "vv_median": float(np.nanmedian(vv)),
        "vh_mean": float(np.nanmean(vh)),
        "vh_std": float(np.nanstd(vh)),
        "vh_median": float(np.nanmedian(vh)),
    }
    feats["vv_vh_ratio_mean"] = feats["vv_mean"] - feats["vh_mean"]  # dB domain: ratio = subtraction
    return feats


def glcm_texture_sar(patch, band="vv", levels=32, distances=(1,), angles=(0,)):
    """GLCM texture on SAR backscatter (speckle-aware: quantize after dB scaling)."""
    band_arr = patch[SAR_BAND_IDX[band]]
    if not IS_DB:
        band_arr = to_db(band_arr)

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


def extract_sar_features(patch):
    """
    Main entry point: given one SAR patch (2, H, W) = [VV, VH], return a
    flat dict of feature_name -> value.

    NOTE: apply speckle filtering (e.g. a Lee or refined-Lee filter, or a
    simple median filter) upstream before feature extraction if your GEE
    export didn't already smooth the data — raw single-look SAR is noisy
    enough that texture features can be dominated by speckle rather than
    genuine field structure.
    """
    feats = {}
    feats.update(backscatter_statistics(patch))
    feats.update(glcm_texture_sar(patch, band="vv"))
    feats.update(glcm_texture_sar(patch, band="vh"))
    return feats
