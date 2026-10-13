"""Plot chosen words and their neighbors in two dimensions."""

import sys

import torch
from yaml_config_manager import load_config

from w2vlab.vectors import WordVectors


def main():
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        raise SystemExit("Error: --config is required. Example: python visualize.py --config configs/visualize.yaml")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config()
    vectors = WordVectors(config.model)
    chosen = [config.words] if isinstance(config.words, str) else config.words
    anchors = [word.lower() for word in chosen]
    for word in anchors:
        vectors.vector(word)  # An OOV word is valid when fastText knows some of its n-grams.
    words = list(anchors)
    for word in anchors:
        words.extend(name for name, _ in vectors.most_similar(word, config.neighbors))
    words = list(dict.fromkeys(words))
    anchor_set = set(anchors)
    if len(words) < 2:
        raise ValueError("choose at least two words")

    matrix = torch.stack([torch.nn.functional.normalize(vectors.vector(word), dim=0)
                          for word in words])
    centered = matrix - matrix.mean(dim=0)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    points = u[:, :2] * s[:2]  # PCA of the selected word vectors

    fig, ax = plt.subplots(figsize=(8, 6))
    for word, (x, y) in zip(words, points.tolist()):
        ax.scatter(x, y, color="tab:orange" if word in anchor_set else "tab:blue")
        ax.annotate(word, (x, y), xytext=(4, 4), textcoords="offset points")
    ax.scatter([], [], color="tab:orange", label="Chosen words")
    ax.scatter([], [], color="tab:blue", label="Neighbors")
    ax.legend()
    ax.margins(0.15)
    ax.set(title="Word vectors (PCA)", xlabel="PC 1", ylabel="PC 2")
    fig.tight_layout()
    fig.savefig(config.output, dpi=150)
    plt.close(fig)
    print(f"Saved {config.output}")


if __name__ == "__main__":
    main()
