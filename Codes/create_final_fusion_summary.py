import os
import pandas as pd

ROOT = r"E:\DIP2026\Codes"

file = os.path.join(
    ROOT, "SAR_Feature_Extraction",
    "ALL_FIELDS_SAR_Optical_Fusion_With_Texture.csv"
)

out = os.path.join(ROOT, "Final_Master_Results")
os.makedirs(out, exist_ok=True)

df = pd.read_csv(file)
df = df[df["SAR_texture_available"] == True].copy()

features = [
    "NDVI_mean",
    "VV_mean",
    "VH_mean",
    "VV_GLCM_contrast",
    "VV_GLCM_homogeneity",
    "VV_GLCM_energy",
    "VV_GLCM_correlation",
    "VH_GLCM_contrast",
    "VH_GLCM_homogeneity",
    "VH_GLCM_energy",
    "VH_GLCM_correlation"
]

summary = df.groupby("field_name")[features].agg(
    ["mean", "std"]
)

summary.columns = [
    f"{feature}_{stat}" for feature, stat in summary.columns
]

summary = summary.reset_index()
summary["matched_observations"] = (
    df.groupby("field_name").size().values
)

summary_file = os.path.join(
    out, "SAR_Optical_Texture_Field_Comparison.csv"
)

summary.to_csv(summary_file, index=False)

print("FINAL FIELD COMPARISON")
print(summary.to_string(index=False))
print("\nSaved:", summary_file)
