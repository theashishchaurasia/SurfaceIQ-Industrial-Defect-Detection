
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.config import load_config, project_path
from src.data.dataset import discover_images, NEUDataset, CLASSES
from src.data.prepare import split_records
from src.models.classifiers import build_classifier


def evaluate_model(name, records, image_size, batch_size, device, output_dir):
    checkpoint_path = output_dir.parent / "models" / f"{name}_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_classifier(name, num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    dataset = NEUDataset(records, image_size=image_size, augment=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )

    metrics = {
        "model": name,
        "evaluation_partition": "validation",
        "validation_images": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            c: report[c] for c in CLASSES
        }
    }

    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{name}_classification_report.csv"
    )

    pd.DataFrame(
        cm, index=CLASSES, columns=CLASSES
    ).to_csv(output_dir / f"{name}_confusion_matrix.csv")

    predictions_df = pd.DataFrame({
        "image": [r["path"] for r in records],
        "actual": [CLASSES[i] for i in y_true],
        "predicted": [CLASSES[i] for i in y_pred],
        "correct": [a == p for a, p in zip(y_true, y_pred)]
    })

    predictions_df.to_csv(
        output_dir / f"{name}_predictions.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{name.upper()} - Validation Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(f"\n{name.upper()} RESULTS")
    print(f"Validation images: {len(y_true)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved results in: {output_dir}")


def main():
    cfg = load_config()
    dataset_cfg = cfg["dataset"]

    seed = cfg.get("project", {}).get("seed", cfg.get("seed", 42))
    image_size = dataset_cfg["image_size"]

    root = project_path(dataset_cfg["root"], cfg)
    records = discover_images(root)
    records = [r for r in records if r["label"] in CLASSES]

    _, validation_records, _ = split_records(
        records,
        seed,
        dataset_cfg["train_fraction"],
        dataset_cfg["val_fraction"]
    )

    output_dir = project_path(cfg["results_dir"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Validation images: {len(validation_records)}")
    print(f"Classes: {CLASSES}")

    for name in ["cnn", "resnet18"]:
        evaluate_model(
            name,
            validation_records,
            image_size,
            dataset_cfg["batch_size"],
            device,
            output_dir
        )

    print("\nBoth classifier evaluations completed successfully.")


if __name__ == "__main__":
    main()
