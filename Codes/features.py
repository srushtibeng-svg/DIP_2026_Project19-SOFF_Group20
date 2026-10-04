"""
features.py - classical hand-crafted features for SAR-Optical Feature Fusion
ECE 501 DIP | Project 19 | Group 20

Every function takes plain numpy arrays (one 2D array per band), so it works
whatever file format YieldSAT uses. Load the bands first (rasterio / numpy),
then pass them in.

Note: Sentinel-2 values are often stored as integers 0-10000. Divide by 10000
to get reflectance (0-1) BEFORE calling the optical functions.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops

EPS = 1e-8


# Optical: spectral indices
def ndvi(nir, red):
    return (nir - red) / (nir + red + EPS)


def ndre(nir, red_edge):
    return (nir - red_edge) / (nir + red_edge + EPS)


def ndmi(nir, swir):
    return (nir - swir) / (nir + swir + EPS)


def evi(nir, red, blue):
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)


# GLCM texture (works for optical and SAR)
def glcm_features(img, levels=32):
    """Contrast, homogeneity, energy, correlation (mean over 4 directions)."""
    img = np.nan_to_num(img.astype(np.float64))
    lo, hi = img.min(), img.max()
    q = np.zeros_like(img, dtype=np.uint8) if hi - lo < EPS else \
        ((img - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    return {p: float(graycoprops(glcm, p).mean())
            for p in ("contrast", "homogeneity", "energy", "correlation")}


# Feature vectors per field/patch
def optical_features(blue, red, red_edge, nir, swir):
    """Return a dict of optical features for one patch (bands as 2D arrays)."""
    idx = {
        "ndvi": ndvi(nir, red),
        "ndre": ndre(nir, red_edge),
        "ndmi": ndmi(nir, swir),
        "evi": evi(nir, red, blue),
    }
    feats = {}
    for name, arr in idx.items():
        feats[f"{name}_mean"] = float(np.nanmean(arr))
        feats[f"{name}_std"] = float(np.nanstd(arr))
    for k, v in glcm_features(idx["ndvi"]).items():
        feats[f"opt_glcm_{k}"] = v
    return feats


def sar_features(vv, vh):
    """vv, vh: backscatter in LINEAR scale (convert from dB: 10**(dB/10))."""
    ratio = vv / (vh + EPS)
    feats = {
        "vv_mean": float(np.nanmean(vv)), "vv_var": float(np.nanvar(vv)),
        "vh_mean": float(np.nanmean(vh)), "vh_var": float(np.nanvar(vh)),
        "vvvh_ratio_mean": float(np.nanmean(ratio)),
    }
    vv_db = 10 * np.log10(np.clip(vv, EPS, None))
    for k, v in glcm_features(vv_db).items():
        feats[f"sar_glcm_{k}"] = v
    return feats


def fuse(optical_feats, sar_feats):
    """Feature-level fusion = concatenate the two dicts (normalise later,
    after the train/test split, using StandardScaler fit on train only)."""
    return {**optical_feats, **sar_feats}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    b = {k: rng.uniform(0.02, 0.5, (64, 64)) for k in
         ("blue", "red", "red_edge", "nir", "swir")}
    o = optical_features(**b)
    s = sar_features(rng.uniform(0.01, 0.3, (64, 64)),
                     rng.uniform(0.005, 0.1, (64, 64)))
    print(len(fuse(o, s)), "fused features")
