from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

root = Path(__file__).resolve().parent
source = root / "SAR_30_features" / "ML_Results_30" / "oof_predictions.csv"
out = root / "SAR_30_features" / "ML_Results_30" / "Confusion_Matrices"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)
labels = ["rapeseed", "wheat"]
combined = []

for (modality, model), group in df.groupby(["modality", "model"]):
    cm = confusion_matrix(
        group["crop_actual"],
        group["crop_predicted"],
        labels=labels
    )

    name = f"{modality}_{model}"
    matrix_df = pd.DataFrame(
        cm,
        index=[f"Actual_{x}" for x in labels],
        columns=[f"Predicted_{x}" for x in labels]
    )
    matrix_df.to_csv(out / f"{name}_confusion_matrix.csv")

    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            combined.append({
                "modality": modality,
                "model": model,
                "actual": actual,
                "predicted": predicted,
                "count": int(cm[i, j])
            })

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{modality} - {model.replace('_', ' ')}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png", dpi=200)
    plt.close(fig)

pd.DataFrame(combined).to_csv(out / "all_confusion_matrices.csv", index=False)

print("Confusion matrices generated successfully.")
print("Output folder:", out)
print("Files created:", len(list(out.iterdir())))
for path in sorted(out.iterdir()):
    print(path.name)
