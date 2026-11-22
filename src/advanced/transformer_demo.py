
import json
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path


VOCAB = [
    "<PAD>", "<CLS>", "<SEP>", "<UNK>",
    "steel", "surface", "defect", "detected", "inspection",
    "crazing", "inclusion", "patches", "pitted", "rolled",
    "scale", "scratches", "cracks", "texture", "damage",
    "severity", "high", "medium", "low", "minor",
    "major", "visible", "image", "metal", "quality",
    "production", "line", "area", "multiple", "small",
    "large", "near", "center", "edge", "observed"
]

WORD_TO_ID = {word: i for i, word in enumerate(VOCAB)}


def tokenize(text):
    words = re.findall(r"[a-z]+", text.lower())
    ids = [1] + [WORD_TO_ID.get(w, 3) for w in words] + [2]
    tokens = ["<CLS>"] + words + ["<SEP>"]
    return tokens, ids


class TransformerReportEncoder(nn.Module):
    def __init__(self, vocab_size, dimension=64, heads=4, max_length=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dimension)
        self.position = nn.Embedding(max_length, dimension)

        self.attention = nn.MultiheadAttention(
            dimension, heads, batch_first=True
        )

        self.norm1 = nn.LayerNorm(dimension)

        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.ReLU(),
            nn.Linear(128, dimension)
        )

        self.norm2 = nn.LayerNorm(dimension)

    def forward(self, ids):
        positions = torch.arange(
            ids.shape[1], device=ids.device
        ).unsqueeze(0)

        x = self.embedding(ids) + self.position(positions)

        attended, weights = self.attention(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )

        x = self.norm1(x + attended)
        x = self.norm2(x + self.feed_forward(x))

        # CLS token provides a fixed-size report representation.
        report_embedding = x[:, 0, :]

        return report_embedding, weights


def main():
    torch.manual_seed(42)

    report = (
        "Steel surface inspection detected scratches and "
        "pitted defects near the center with high severity"
    )

    tokens, ids = tokenize(report)
    input_tensor = torch.tensor([ids], dtype=torch.long)

    model = TransformerReportEncoder(len(VOCAB))
    model.eval()

    with torch.inference_mode():
        embedding, attention = model(input_tensor)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    embedding_path = output_dir / "transformer_report_embedding.json"
    embedding_path.write_text(
        json.dumps({
            "report": report,
            "tokens": tokens,
            "embedding_dimension": embedding.shape[1],
            "embedding": embedding[0].tolist(),
            "model": "Custom PyTorch Transformer encoder",
            "training_status": "Untrained educational prototype"
        }, indent=2),
        encoding="utf-8"
    )

    # Average attention across heads for the first and only layer.
    attention_map = attention[0].mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 7))
    plt.imshow(attention_map, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Tokens receiving attention")
    plt.ylabel("Tokens producing attention")
    plt.title("Transformer Self-Attention Map")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(output_dir / "transformer_attention.png", dpi=180)
    plt.close()

    print("TRANSFORMER DEMONSTRATION COMPLETED")
    print(f"Input tokens: {tokens}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Attention shape: {tuple(attention.shape)}")
    print(f"Embedding saved: {embedding_path}")
    print(f"Attention map saved: {output_dir / 'transformer_attention.png'}")


if __name__ == "__main__":
    main()
