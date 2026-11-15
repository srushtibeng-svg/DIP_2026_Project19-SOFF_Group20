import os
import numpy as np
import pandas as pd
import rasterio

ROOT = r"E:\DIP2026\Codes"
TIFF_ROOT = os.path.join(ROOT, "S1_selected_orbit_downloads")
STATS_ROOT = os.path.join(ROOT, "SAR_timeseries_statistics")
OUT = os.path.join(ROOT, "SAR_Feature_Extraction")

os.makedirs(OUT, exist_ok=True)


def get_stats(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]

    if len(values) == 0:
        return {}

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_median": np.median(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_p25": np.percentile(values, 25),
        f"{prefix}_p75": np.percentile(values, 75),
        f"{prefix}_valid_pixels": len(values)
    }


def glcm_features(array):
    array = np.asarray(array, dtype=float)
    valid = np.isfinite(array) & (array != 0)

    if valid.sum() < 20:
        return {
            "contrast": np.nan,
            "homogeneity": np.nan,
            "energy": np.nan,
            "correlation": np.nan
        }

    # Fixed dB range: -50 to 0, quantized into 32 levels
    quantized = np.clip(
        np.floor((array + 50) / 50 * 31), 0, 31
    ).astype(int)

    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    matrices = []

    for dy, dx in offsets:
        h, w = array.shape

        y1 = max(0, -dy)
        y2 = min(h, h - dy)
        x1 = max(0, -dx)
        x2 = min(w, w - dx)

        a = quantized[y1:y2, x1:x2]
        b = quantized[y1+dy:y2+dy, x1+dx:x2+dx]

        ma = valid[y1:y2, x1:x2]
        mb = valid[y1+dy:y2+dy, x1+dx:x2+dx]

        pair_valid = ma & mb
        if not pair_valid.any():
            continue

        i = a[pair_valid]
        j = b[pair_valid]

        matrix = np.zeros((32, 32), dtype=float)

        np.add.at(matrix, (i, j), 1)
        np.add.at(matrix, (j, i), 1)

        if matrix.sum() > 0:
            matrix /= matrix.sum()
            matrices.append(matrix)

    if not matrices:
        return {
            "contrast": np.nan,
            "homogeneity": np.nan,
            "energy": np.nan,
            "correlation": np.nan
        }

    p = np.mean(matrices, axis=0)

    levels = np.arange(32)
    i, j = np.meshgrid(levels, levels, indexing="ij")

    contrast = np.sum(p * (i - j) ** 2)
    homogeneity = np.sum(p / (1 + (i - j) ** 2))
    energy = np.sqrt(np.sum(p ** 2))

    px = p.sum(axis=1)
    py = p.sum(axis=0)

    mux = np.sum(levels * px)
    muy = np.sum(levels * py)

    sigx = np.sqrt(np.sum((levels - mux) ** 2 * px))
    sigy = np.sqrt(np.sum((levels - muy) ** 2 * py))

    if sigx > 0 and sigy > 0:
        correlation = np.sum(
            p * (i - mux) * (j - muy)
        ) / (sigx * sigy)
    else:
        correlation = np.nan

    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation
    }


all_rows = []

field_folders = sorted([
    f for f in os.listdir(TIFF_ROOT)
    if os.path.isdir(os.path.join(TIFF_ROOT, f))
])

for field in field_folders:
    print("\nProcessing:", field)

    stats_file = os.path.join(
        STATS_ROOT, field + "_SAR_statistics.csv"
    )

    if not os.path.exists(stats_file):
        print("Statistics CSV missing. Skipping.")
        continue

    stats_df = pd.read_csv(stats_file)

    field_dir = os.path.join(TIFF_ROOT, field)

    for _, meta in stats_df.iterrows():
        batch = str(meta["batch_name"])
        tif_path = os.path.join(field_dir, batch + ".tif")

        if not os.path.exists(tif_path):
            print("Missing TIFF:", batch)
            continue

        vv_band = int(meta["VV_band"])
        vh_band = int(meta["VH_band"])

        with rasterio.open(tif_path) as src:
            if vv_band > src.count or vh_band > src.count:
                print(
                    "Invalid band reference:",
                    batch, vv_band, vh_band,
                    "available:", src.count
                )
                continue

            vv = src.read(vv_band).astype(float)
            vh = src.read(vh_band).astype(float)

            if src.nodata is not None:
                vv[vv == src.nodata] = np.nan
                vh[vh == src.nodata] = np.nan

        vv_valid = vv[np.isfinite(vv) & (vv != 0)]
        vh_valid = vh[np.isfinite(vh) & (vh != 0)]

        row = {
            "field_name": field,
            "batch_name": batch,
            "acquisition_date": meta["acquisition_date"],
            "image_id": meta["image_id"],
            "orbit_pass": meta["orbit_pass"],
            "relative_orbit": meta["relative_orbit"],
            "platform": meta["platform"],
            "VV_band": vv_band,
            "VH_band": vh_band,
        }

        row.update(get_stats(vv_valid, "VV"))
        row.update(get_stats(vh_valid, "VH"))

        vv_texture = glcm_features(vv)
        vh_texture = glcm_features(vh)

        for key, value in vv_texture.items():
            row["VV_GLCM_" + key] = value

        for key, value in vh_texture.items():
            row["VH_GLCM_" + key] = value

        all_rows.append(row)

        print(
            meta["acquisition_date"],
            "| VV:", len(vv_valid),
            "| VH:", len(vh_valid)
        )


if not all_rows:
    raise RuntimeError("No SAR features extracted. Check paths and band metadata.")

result = pd.DataFrame(all_rows)

acq_file = os.path.join(OUT, "SAR_Acquisition_Level_Features.csv")
result.to_csv(acq_file, index=False)

numeric_cols = result.select_dtypes(include=np.number).columns.tolist()

summary = result.groupby("field_name")[numeric_cols].agg(
    ["mean", "std", "min", "max"]
)

summary.columns = [
    f"{col}_{stat}" for col, stat in summary.columns
]

summary = summary.reset_index()

summary_file = os.path.join(OUT, "SAR_Field_Level_Features.csv")
summary.to_csv(summary_file, index=False)

print("\n========================================")
print("SAR FEATURE EXTRACTION COMPLETED")
print("Acquisition rows:", len(result))
print("Fields:", result["field_name"].nunique())
print("Acquisition CSV:", acq_file)
print("Field summary CSV:", summary_file)
print("\nRows per field:")
print(result.groupby("field_name").size())
print("========================================")
