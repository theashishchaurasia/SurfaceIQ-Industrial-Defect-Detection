
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA

from src.config import load_config, project_path
from src.data.dataset import discover_images, CLASSES
from src.data.prepare import split_records
from src.generative.vae import TinyVAE, vae_loss


class GrayscaleDefectDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = CLASSES.index(record["label"])
        return image, label


def main():
    cfg = load_config()
    seed = cfg.get("project", {}).get("seed", cfg.get("seed", 42))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    root = project_path(cfg["dataset"]["root"], cfg)
    output_dir = project_path(cfg["results_dir"], cfg)
    model_dir = project_path(cfg["models"]["checkpoint_dir"], cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    records = [
        r for r in discover_images(root)
        if r["label"] in CLASSES
    ]

    train_records, val_records, _ = split_records(
        records,
        seed,
        cfg["dataset"]["train_fraction"],
        cfg["dataset"]["val_fraction"]
    )

    # Keep this demonstration small enough for CPU training.
    random.shuffle(train_records)
    random.shuffle(val_records)

    train_records = train_records[:600]
    val_records = val_records[:180]

    train_loader = DataLoader(
        GrayscaleDefectDataset(train_records),
        batch_size=64,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        GrayscaleDefectDataset(val_records),
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cpu")
    model = TinyVAE(latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    history = []

    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(val_records)}")
    print(f"Device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = model(images)
            loss = vae_loss(reconstructed, images, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_records)
        history.append({
            "epoch": epoch + 1,
            "training_loss_per_image": average_loss
        })

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss per image: {average_loss:.4f}"
        )

    # Evaluate reconstruction on held-out validation images.
    model.eval()
    all_mu = []
    all_labels = []
    validation_loss = 0
    example_images = None
    example_reconstructions = None

    with torch.inference_mode():
        for images, labels in val_loader:
            reconstructed, mu, logvar = model(images)

            validation_loss += vae_loss(
                reconstructed, images, mu, logvar
            ).item()

            all_mu.append(mu.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

            if example_images is None:
                example_images = images[:8].cpu()
                example_reconstructions = reconstructed[:8].cpu()

    latent_vectors = np.concatenate(all_mu, axis=0)

    # Save model checkpoint.
    torch.save({
        "state_dict": model.state_dict(),
        "latent_dim": 8,
        "input_shape": [1, 28, 28],
        "epochs": epochs,
        "training_images": len(train_records),
        "validation_images": len(val_records)
    }, model_dir / "tiny_vae.pt")

    # Save reconstruction comparison.
    fig, axes = plt.subplots(2, 8, figsize=(13, 4))

    for i in range(8):
        axes[0, i].imshow(example_images[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(example_reconstructions[i, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    fig.suptitle("VAE: Original vs Reconstructed Steel Defect Images")
    fig.tight_layout()
    fig.savefig(output_dir / "vae_reconstructions.png", dpi=180)
    plt.close(fig)

    # Reduce 8-dimensional latent means to 2D for visualisation.
    latent_2d = PCA(n_components=2).fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=all_labels,
        cmap="tab10",
        alpha=0.75
    )
    plt.colorbar(scatter, ticks=range(len(CLASSES)), label="Defect class")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("VAE Latent Space - Validation Images")
    plt.tight_layout()
    plt.savefig(output_dir / "vae_latent_space.png", dpi=180)
    plt.close()

    metrics = {
        "model": "TinyVAE",
        "latent_dimension": 8,
        "input_size": "28x28 grayscale",
        "training_images": len(train_records),
        "validation_images": len(val_records),
        "epochs": epochs,
        "validation_loss_per_image": validation_loss / len(val_records),
        "training_history": history,
        "purpose": "Educational image reconstruction and latent-space demonstration"
    }

    (output_dir / "vae_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nVAE DEMONSTRATION COMPLETED")
    print(f"Checkpoint: {model_dir / 'tiny_vae.pt'}")
    print(f"Reconstructions: {output_dir / 'vae_reconstructions.png'}")
    print(f"Latent space: {output_dir / 'vae_latent_space.png'}")
    print(f"Metrics: {output_dir / 'vae_metrics.json'}")


if __name__ == "__main__":
    main()
