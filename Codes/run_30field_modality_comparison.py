import os
import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

base = "SAR_30_features"
out = os.path.join(base, "ML_Results_30")
os.makedirs(out, exist_ok=True)

optical_features = [
    "ndvi_mean", "ndvi_std", "ndvi_max",
    "ndre_mean", "ndre_std", "ndre_max",
    "ndmi_mean", "ndmi_std", "ndmi_max",
    "evi_mean", "evi_std", "evi_max"
]

sar_features = [
    "VV_mean_temporal_mean", "VV_mean_temporal_std",
    "VV_mean_temporal_min", "VV_mean_temporal_max",
    "VH_mean_temporal_mean", "VH_mean_temporal_std",
    "VH_mean_temporal_min", "VH_mean_temporal_max",
    "VV_stdDev_temporal_mean", "VV_stdDev_temporal_std",
    "VV_stdDev_temporal_min", "VV_stdDev_temporal_max",
    "VH_stdDev_temporal_mean", "VH_stdDev_temporal_std",
    "VH_stdDev_temporal_min", "VH_stdDev_temporal_max"
]

datasets = {
    "Optical": ("Optical_30_Fields.csv", optical_features),
    "SAR": ("SAR_30_Fields.csv", sar_features),
    "Fusion": ("Optical_SAR_Fusion_30_Fields.csv",
               optical_features + sar_features)
}

models = {
    "Random_Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1, gamma="scale",
            class_weight="balanced")
    )
}

summary = []
fold_results = []
all_predictions = []
labels = ["rapeseed", "wheat"]

for modality, (filename, features) in datasets.items():
    path = os.path.join(base, filename)
    df = pd.read_csv(path)

    required = features + [
        "field_shared_name", "crop", "farm_identifier"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{modality}: Missing columns: {missing}")

    if df["field_shared_name"].nunique() != 30:
        raise ValueError(f"{modality}: Expected exactly 30 unique fields")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(f"{modality}: Missing/non-numeric feature values")

    y = df["crop"].astype(str).str.lower()
    groups = df["farm_identifier"].astype(str)

    print(f"\n{'='*55}")
    print(f"{modality}: {len(features)} features")
    print(f"Fields: {len(df)}, Farms: {groups.nunique()}")
    print(f"Crop counts: {y.value_counts().to_dict()}")

    logo = LeaveOneGroupOut()

    for model_name, model in models.items():
        oof = np.empty(len(df), dtype=object)
        fold_number = 0

        for train_idx, test_idx in logo.split(X, y, groups):
            fold_number += 1
            fitted = model
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = fitted.predict(X.iloc[test_idx])
            oof[test_idx] = pred

            actual = y.iloc[test_idx]
            fold_results.append({
                "modality": modality,
                "model": model_name,
                "test_farm": groups.iloc[test_idx].iloc[0],
                "test_fields": len(test_idx),
                "accuracy": accuracy_score(actual, pred),
                "balanced_accuracy": balanced_accuracy_score(actual, pred),
                "macro_f1": f1_score(
                    actual, pred, labels=labels,
                    average="macro", zero_division=0
                )
            })

        accuracy = accuracy_score(y, oof)
        balanced = balanced_accuracy_score(y, oof)
        precision = precision_score(
            y, oof, labels=labels, average="macro", zero_division=0
        )
        recall = recall_score(
            y, oof, labels=labels, average="macro", zero_division=0
        )
        f1 = f1_score(
            y, oof, labels=labels, average="macro", zero_division=0
        )
        cm = confusion_matrix(y, oof, labels=labels)

        summary.append({
            "modality": modality,
            "model": model_name,
            "features": len(features),
            "fields": len(df),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        })

        for i in range(len(df)):
            all_predictions.append({
                "field_shared_name": df.iloc[i]["field_shared_name"],
                "crop_actual": y.iloc[i],
                "farm_identifier": groups.iloc[i],
                "modality": modality,
                "model": model_name,
                "crop_predicted": oof[i]
            })

        print(f"\n{model_name}")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced:.4f}")
        print(f"Macro Precision:   {precision:.4f}")
        print(f"Macro Recall:      {recall:.4f}")
        print(f"Macro F1:          {f1:.4f}")
        print("Confusion Matrix (rows=actual, columns=predicted):")
        print(pd.DataFrame(
            cm, index=labels, columns=labels
        ))

pd.DataFrame(summary).to_csv(
    os.path.join(out, "summary_metrics.csv"), index=False
)
pd.DataFrame(fold_results).to_csv(
    os.path.join(out, "fold_metrics.csv"), index=False
)
pd.DataFrame(all_predictions).to_csv(
    os.path.join(out, "oof_predictions.csv"), index=False
)

print("\nAll experiments completed.")
print("\nSummary:")
print(pd.DataFrame(summary).round(4).to_string(index=False))
print("\nResults saved in:", out)
