
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
DATA_FILE = "germany_optical_features_120.csv"
OUT_DIR = "Optical_ML_Results"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

FEATURES = [
    "ndvi_mean", "ndvi_std", "ndvi_max",
    "ndre_mean", "ndre_std", "ndre_max",
    "ndmi_mean", "ndmi_std", "ndmi_max",
    "evi_mean", "evi_std", "evi_max"
]

required = FEATURES + ["crop", "farm_identifier", "field_shared_name"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).reset_index(drop=True)

X = df[FEATURES].astype(float)
y = df["crop"].astype(str).str.lower()
groups = df["farm_identifier"].astype(str)

print("Dataset:", df.shape)
print("\nCrop counts:\n", y.value_counts())
print("\nFields per farm:\n", groups.value_counts())
print("\nFeatures used:", len(FEATURES))

# --------------------------------------------------
# 2. Models
# --------------------------------------------------
models = {
    "Random_Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    ),

    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])
}

# --------------------------------------------------
# 3. Leave-One-Farm-Out evaluation
# --------------------------------------------------
logo = LeaveOneGroupOut()

all_metrics = []
all_predictions = []

for model_name, model in models.items():
    print(f"\n========== {model_name} ==========")

    y_true_all = []
    y_pred_all = []

    for fold_no, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        farm_name = groups.iloc[test_idx].iloc[0]

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        rec = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        print(
            f"Fold {fold_no} | Test farm: {farm_name} | "
            f"Fields: {len(test_idx)} | Accuracy: {acc:.4f} | "
            f"Macro F1: {f1:.4f}"
        )

        all_metrics.append({
            "model": model_name,
            "fold": fold_no,
            "test_farm": farm_name,
            "test_fields": len(test_idx),
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1
        })

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(predictions.tolist())

        for idx, actual, predicted in zip(
            test_idx, y_test.tolist(), predictions.tolist()
        ):
            all_predictions.append({
                "model": model_name,
                "field_shared_name": df.iloc[idx]["field_shared_name"],
                "farm_identifier": groups.iloc[idx],
                "actual_crop": actual,
                "predicted_crop": predicted,
                "correct": actual == predicted
            })

    # Overall out-of-fold metrics
    print("\nPooled out-of-fold results:")
    print("Accuracy:", round(accuracy_score(y_true_all, y_pred_all), 4))
    print(
        "Macro Precision:",
        round(precision_score(
            y_true_all, y_pred_all, average="macro", zero_division=0
        ), 4)
    )
    print(
        "Macro Recall:",
        round(recall_score(
            y_true_all, y_pred_all, average="macro", zero_division=0
        ), 4)
    )
    print(
        "Macro F1:",
        round(f1_score(
            y_true_all, y_pred_all, average="macro", zero_division=0
        ), 4)
    )

    print("\nClassification report:")
    print(classification_report(
        y_true_all, y_pred_all, zero_division=0
    ))

    labels = ["rapeseed", "wheat"]
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    print("Confusion matrix (rapeseed, wheat):")
    print(cm)

    # Save confusion matrix image
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set(
        xticks=range(2),
        yticks=range(2),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted crop",
        ylabel="Actual crop",
        title=f"{model_name} - Leave-One-Farm-Out"
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, f"{model_name}_confusion_matrix.png"),
        dpi=200
    )
    plt.close(fig)

# --------------------------------------------------
# 4. Save results
# --------------------------------------------------
metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.DataFrame(all_predictions)

metrics_df.to_csv(
    os.path.join(OUT_DIR, "farmwise_fold_metrics.csv"), index=False
)
predictions_df.to_csv(
    os.path.join(OUT_DIR, "out_of_fold_predictions.csv"), index=False
)

summary = metrics_df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_macro_f1=("macro_f1", "mean"),
    std_macro_f1=("macro_f1", "std"),
    mean_macro_precision=("macro_precision", "mean"),
    mean_macro_recall=("macro_recall", "mean")
).reset_index()

summary.to_csv(
    os.path.join(OUT_DIR, "model_comparison_summary.csv"), index=False
)

print("\n========== MODEL COMPARISON ==========")
print(summary.round(4).to_string(index=False))

# --------------------------------------------------
# 5. Train final models on all 120 fields
# --------------------------------------------------
for model_name, model in models.items():
    model.fit(X, y)
    joblib.dump(
        model,
        os.path.join(OUT_DIR, f"{model_name}_final_model.joblib")
    )

with open(os.path.join(OUT_DIR, "experiment_info.json"), "w") as f:
    json.dump({
        "dataset": DATA_FILE,
        "number_of_fields": len(df),
        "features": FEATURES,
        "classes": sorted(y.unique().tolist()),
        "validation": "Leave-One-Farm-Out",
        "number_of_farms": int(groups.nunique()),
        "note": "Final saved models are fitted on all 120 fields. "
                "Evaluation metrics come from farm-wise held-out predictions."
    }, f, indent=4)

print("\nDONE! Results saved in:", OUT_DIR)
