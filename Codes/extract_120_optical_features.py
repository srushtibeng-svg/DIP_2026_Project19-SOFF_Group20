import zipfile
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from pathlib import PurePosixPath

ZIP_PATH = r"D:\downloads\Raw.zip"
INPUT_CSV = "germany_subset_120.csv"
OUTPUT_CSV = "germany_optical_features_120.csv"

BANDS = {"B02": 2, "B04": 4, "B05": 5, "B08": 8, "B11": 11}
VALID_SCL = (4, 5)
INDEX_NAMES = ["ndvi", "ndre", "ndmi", "evi"]


def date_from_name(name):
    m = re.search(r"(\d{8})\.tif$", name, re.I)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d")


def read_raster_from_zip(z, name):
    with MemoryFile(z.read(name)) as mem:
        with mem.open() as ds:
            return ds.read()


def image_indices(z, s2_name, scl_name):
    s2 = read_raster_from_zip(z, s2_name)
    scl = read_raster_from_zip(z, scl_name)[0]

    if s2.shape[1:] != scl.shape:
        raise ValueError("S2 and SCL dimensions do not match")

    valid = np.isin(scl, VALID_SCL)

    bands = {}
    for band, number in BANDS.items():
        arr = s2[number - 1].astype(np.float32) / 10000.0
        valid_band = np.isfinite(arr) & (arr > 0)
        valid &= valid_band
        bands[band] = arr

    if valid.sum() == 0:
        return None

    b02 = bands["B02"]
    b04 = bands["B04"]
    b05 = bands["B05"]
    b08 = bands["B08"]
    b11 = bands["B11"]

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b08 - b04) / (b08 + b04)
        ndre = (b08 - b05) / (b08 + b05)
        ndmi = (b08 - b11) / (b08 + b11)
        evi_den = b08 + 6*b04 - 7.5*b02 + 1
        evi = 2.5 * (b08 - b04) / evi_den

    indices = {"ndvi": ndvi, "ndre": ndre, "ndmi": ndmi, "evi": evi}
    result = {}

    for key, arr in indices.items():
        good = valid & np.isfinite(arr)
        if good.sum():
            result[key] = float(np.mean(arr[good]))
        else:
            result[key] = np.nan

    result["valid_pixels"] = int(valid.sum())
    result["total_pixels"] = int(valid.size)
    return result


def main():
    df = pd.read_csv(INPUT_CSV)
    df["sowing_dt"] = pd.to_datetime(df["seeding_date"], format="%d.%m.%Y")
    df["harvest_dt"] = pd.to_datetime(df["harvesting_date"], format="%d.%m.%Y")

    # Index ZIP entries once, grouped by field and acquisition date.
    s2_files = {}
    scl_files = {}

    print("Indexing ZIP contents...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in z.namelist():
            if not name.lower().endswith(".tif"):
                continue

            parts = PurePosixPath(name).parts
            if len(parts) < 4:
                continue

            field = parts[1]
            if parts[2] not in ("s2_images", "scl_masks"):
                continue

            dt = date_from_name(name)
            if dt is None:
                continue

            target = s2_files if parts[2] == "s2_images" else scl_files
            target.setdefault(field, {})[dt] = name

        print("Fields with S2:", len(s2_files))
        print("Fields with SCL:", len(scl_files))

        output_rows = []

        for i, row in df.iterrows():
            field = row["field_shared_name"]
            start = row["sowing_dt"]
            end = row["harvest_dt"]

            s2_dates = s2_files.get(field, {})
            scl_dates = scl_files.get(field, {})

            matched_dates = sorted(
                d for d in s2_dates
                if d in scl_dates and start <= d <= end
            )

            series = {k: [] for k in INDEX_NAMES}
            valid_pixel_ratios = []

            for dt in matched_dates:
                try:
                    result = image_indices(z, s2_dates[dt], scl_dates[dt])
                    if result is None:
                        continue

                    for key in INDEX_NAMES:
                        if np.isfinite(result[key]):
                            series[key].append(result[key])

                    valid_pixel_ratios.append(
                        result["valid_pixels"] / result["total_pixels"]
                    )
                except Exception as e:
                    print("Image error:", field, dt.date(), str(e))

            features = {
                "field_shared_name": field,
                "crop": row["crop"],
                "year": int(row["year"]),
                "farm_identifier": row["farm_identifier"],
                "n_matched_images": len(matched_dates),
                "n_valid_dates": len(valid_pixel_ratios),
                "mean_valid_pixel_ratio": (
                    float(np.mean(valid_pixel_ratios))
                    if valid_pixel_ratios else np.nan
                )
            }

            for key in INDEX_NAMES:
                values = np.asarray(series[key], dtype=float)
                values = values[np.isfinite(values)]

                features[key + "_mean"] = float(values.mean()) if len(values) else np.nan
                features[key + "_std"] = float(values.std()) if len(values) else np.nan
                features[key + "_max"] = float(values.max()) if len(values) else np.nan

            output_rows.append(features)
            print(
                f"[{i+1}/{len(df)}] {field}: "
                f"{len(matched_dates)} matched, "
                f"{features['n_valid_dates']} valid dates"
            )

        result_df = pd.DataFrame(output_rows)
        result_df.to_csv(OUTPUT_CSV, index=False)

    print("\n===== EXTRACTION SUMMARY =====")
    print("Rows:", len(result_df))
    print("Features:", len(result_df.columns))
    print(result_df.groupby("crop")["n_valid_dates"].describe().round(2))
    print("\nMissing feature values:")
    print(result_df.isna().sum().to_string())
    print("\nSaved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
