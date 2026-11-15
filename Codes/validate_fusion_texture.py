import os
import pandas as pd

file = r"E:\DIP2026\Codes\SAR_Feature_Extraction\ALL_FIELDS_SAR_Optical_Fusion_With_Texture.csv"

df = pd.read_csv(file)

texture_cols = [
    c for c in df.columns if "GLCM" in c
]

matched = df[df["SAR_texture_available"] == True]
unmatched = df[df["SAR_texture_available"] == False]

print("Dataset shape:", df.shape)
print("Matched rows:", len(matched))
print("Unmatched rows:", len(unmatched))

print("\nTexture missing values in matched rows:")
print(matched[texture_cols].isna().sum())

print("\nTexture missing values in unmatched rows:")
print(unmatched[texture_cols].isna().sum())

print("\nPer-field summary:")
print(
    df.groupby("field_name").agg(
        total_rows=("NDVI_mean", "size"),
        matched=("SAR_texture_available", "sum"),
        avg_NDVI=("NDVI_mean", "mean")
    ).to_string()
)

print("\nAll texture columns:")
print(texture_cols)
