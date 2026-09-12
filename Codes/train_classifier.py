"""
train_classifier.py
--------------------
End-to-end: load features -> train/test split -> train RF/SVM for
optical-only, SAR-only, and fused settings -> report accuracy, F1,
confusion matrix for each, and save comparison plots to Results/.

Expects a feature CSV already built (e.g. via optical_features.py +
sar_features.py + fusion.build_feature_dataframe), OR wire this up to
build features on the fly from raw patches — see the __main__ block for
both options.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from fusion import split_modalities, scale_features

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "Results")


def train_and_eval(X_train, X_test, y_train, y_test, model_name="rf"):
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    elif model_name == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    else:
        raise ValueError(model_name)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": clf,
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "report": report,
    }


def save_confusion_matrix(cm, labels, title, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_comparison(df, model_name="rf", test_size=0.25, random_state=42):
    """
    df: wide feature DataFrame (see fusion.build_feature_dataframe)
    Returns a dict of results for 'optical', 'sar', 'fused'.
    """
    X_opt, X_sar, X_fused, y, opt_cols, sar_cols = split_modalities(df)
    class_labels = sorted(pd.unique(y))

    settings = {"optical": X_opt, "sar": X_sar, "fused": X_fused}
    results = {}

    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "graphs"), exist_ok=True)

    for name, X in settings.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test, _ = scale_features(X_train, X_test)
        res = train_and_eval(X_train, X_test, y_train, y_test, model_name=model_name)
        results[name] = res

        save_confusion_matrix(
            res["confusion_matrix"], class_labels,
            title=f"{name.capitalize()} — {model_name.upper()} confusion matrix",
            out_path=os.path.join(RESULTS_DIR, "tables", f"confusion_{name}_{model_name}.png"),
        )
        print(f"[{name}] accuracy={res['accuracy']:.3f}  macro-F1={res['f1_macro']:.3f}")

    # comparison bar chart
    plt.figure(figsize=(6, 4))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["f1_macro"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width / 2, accs, width, label="Accuracy")
    plt.bar(x + width / 2, f1s, width, label="Macro F1")
    plt.xticks(x, [n.capitalize() for n in names])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Optical vs SAR vs Fused ({model_name.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "graphs", f"comparison_{model_name}.png"), dpi=150)
    plt.close()

    # metrics table
    summary = pd.DataFrame({
        "setting": names,
        "accuracy": accs,
        "macro_f1": f1s,
    })
    summary.to_csv(os.path.join(RESULTS_DIR, "tables", f"metrics_summary_{model_name}.csv"), index=False)

    return results


if __name__ == "__main__":
    # ---- Option A: load a pre-built feature CSV ----
    # df = pd.read_csv("features_wide.csv")

    # ---- Option B: build directly from patches (see optical_features.py /
    # sar_features.py / utils.py to fill in the loading loop), then:
    # from fusion import build_feature_dataframe
    # df = build_feature_dataframe(records)

    raise SystemExit(
        "Fill in feature loading (Option A or B above) before running. "
        "Once df is ready, call run_comparison(df, model_name='rf') and "
        "run_comparison(df, model_name='svm') to compare both classifiers."
    )
