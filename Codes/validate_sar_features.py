import os
import numpy as np
import pandas as pd

ROOT = r"E:\DIP2026\Codes"

acq = pd.read_csv(
    os.path.join(ROOT, "SAR_Feature_Extraction",
                 "SAR_Acquisition_Level_Features.csv")
)

stats_dir = os.path.join(ROOT, "SAR_timeseries_statistics")

checks = []

for field in acq["field_name"].unique():
    original_file = os.path.join(
        stats_dir, field + "_SAR_statistics.csv"
    )

    original = pd.read_csv(original_file)
    extracted = acq[acq["field_name"] == field]

    merged = original.merge(
        extracted,
        on=["batch_name", "acquisition_date", "image_id"],
        suffixes=("_original", "_extracted")
    )

    for pol in ["VV", "VH"]:
        diff = (
            merged[f"{pol}_mean_original"]
            - merged[f"{pol}_mean_extracted"]
        ).abs()

        checks.append({
            "field": field,
            "polarization": pol,
            "matched_rows": len(merged),
            "max_mean_difference": diff.max(),
            "mean_match": bool(np.allclose(
                merged[f"{pol}_mean_original"],
                merged[f"{pol}_mean_extracted"],
                atol=0.0001
            ))
        })

print("\nMEAN VALIDATION")
print(pd.DataFrame(checks).to_string(index=False))

texture_cols = [
    c for c in acq.columns if "GLCM" in c
]

print("\nGLCM missing values:")
print(acq[texture_cols].isna().sum())

print("\nValid pixel count:")
print(acq[["VV_valid_pixels", "VH_valid_pixels"]].describe())

print("\nDuplicate acquisition records:",
      acq.duplicated(
          subset=["field_name", "acquisition_date", "image_id"]
      ).sum())
