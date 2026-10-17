import pandas as pd

germany_file = r"D:\downloads\Germany_Dataset.csv"
uruguay_file = r"D:\downloads\Uruguay_Dataset.csv"

files = {
    "Germany": germany_file,
    "Uruguay": uruguay_file
}

for country, file_path in files.items():
    print("\nDataset:", country)

    df = pd.read_csv(file_path)

    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    if "crop" in df.columns:
        print("\nCrop labels:")
        print(df["crop"].value_counts())
