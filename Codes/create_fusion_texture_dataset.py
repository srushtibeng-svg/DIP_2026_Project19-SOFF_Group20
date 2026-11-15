import os
import pandas as pd

ROOT = r"E:\DIP2026\Codes"

sar_file = os.path.join(
    ROOT, "SAR_Feature_Extraction",
    "SAR_Acquisition_Level_Features.csv"
)

fusion_dir = os.path.join(ROOT, "SAR_Optical_Fusion")
out_dir = os.path.join(ROOT, "SAR_Feature_Extraction")
os.makedirs(out_dir, exist_ok=True)

sar = pd.read_csv(sar_file)
sar["acquisition_date"] = pd.to_datetime(
    sar["acquisition_date"]
).dt.strftime("%Y-%m-%d")

field_map = {
    "Germany_DUP3_farm1_field55_rapeseed_2019":
        "Germany_Rapeseed_2019",

    "Germany_DUP3_farm1_field58_wheat_2019":
        "Germany_Wheat_2019",

    "Uruguay_DUP4_farm2_field29_soybean_2021":
        "Uruguay_Soybean_2021",

    "Uruguay_DUP4_farm7_field178_soybean_2020":
        "Uruguay_Soybean_2020"
}

texture_cols = [
    c for c in sar.columns
    if "GLCM" in c
]

sar_features = [
    "VV_mean", "VH_mean",
    "VV_std", "VH_std"
] + texture_cols

all_fusion = []

for sar_field, fusion_name in field_map.items():

    filename = fusion_name + "_SAR_Optical_Fusion.csv"
    fusion_path = os.path.join(fusion_dir, filename)

    if not os.path.exists(fusion_path):
        print("Missing fusion file:", filename)
        continue

    fusion = pd.read_csv(fusion_path)
    fusion["SAR_date"] = pd.to_datetime(
        fusion["SAR_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    field_sar = sar[sar["field_name"] == sar_field]

    # Average multiple SAR acquisitions on the same date
    daily_sar = (
        field_sar.groupby("acquisition_date")[sar_features]
        .mean()
        .reset_index()
        .rename(columns={"acquisition_date": "SAR_date"})
    )

    # Avoid matching unmatched rows with missing dates
    matched = fusion["SAR_date"].notna()

    fusion["SAR_texture_available"] = False

    fusion.loc[matched, "SAR_texture_available"] = (
        fusion.loc[matched, "SAR_date"].isin(
            daily_sar["SAR_date"]
        )
    )

    fusion = fusion.merge(
        daily_sar,
        on="SAR_date",
        how="left",
        suffixes=("", "_from_SAR_features")
    )

    fusion["country"] = (
        "Germany" if "Germany" in fusion_name else "Uruguay"
    )

    all_fusion.append(fusion)

    print(
        fusion_name,
        "| total:", len(fusion),
        "| texture matched:",
        fusion["SAR_texture_available"].sum()
    )

if not all_fusion:
    raise RuntimeError("No fusion files were processed.")

result = pd.concat(all_fusion, ignore_index=True)

output_file = os.path.join(
    out_dir, "ALL_FIELDS_SAR_Optical_Fusion_With_Texture.csv"
)

result.to_csv(output_file, index=False)

print("\n====================================")
print("SAR-OPTICAL TEXTURE FUSION COMPLETED")
print("Total rows:", len(result))
print("Fields:", result["field_name"].nunique())
print("Texture matched:", result["SAR_texture_available"].sum())
print("Output:", output_file)
print("\nRows by field:")
print(result.groupby("field_name").size())
print("====================================")
