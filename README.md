# w2vlab

A small PyTorch word2vec implementation for the course Elaborazione del Linguaggio Naturale (NLP) at UKE. It creates word/context pairs, trains two embedding tables with skip-gram or CBOW, and saves the input table as word vectors. You can compare negative sampling with full softmax in the same model.

Run everything from the repository root. Install once:

```bash
python -m pip install -e .
```

Every script **requires** `--config`. The linked YAML files are examples to edit and reuse; nothing is loaded implicitly. [yaml_config_manager](https://pypi.org/project/yaml-config-manager/) also accepts overrides after the config path, such as `--training.epochs 2`.

## 1. Download data and train

[configs/download.yaml](configs/download.yaml) selects Sicilian Wikipedia (`language: scn`), the output `scnwiki.txt`, and the article limit. `articles: -1` means the full dump. For a shorter first run:

```bash
python download.py --config configs/download.yaml --articles 1000
```

[configs/train.yaml](configs/train.yaml) reads `scnwiki.txt`, uses the first 200,000 tokens (`data.max_tokens`), and trains skip-gram with negative sampling for five epochs. It saves `vectors.pt`:

```bash
python train.py --config configs/train.yaml
```

Training prints an example pair, tensor shapes, progress, and loss. Rare words share `<unk>`. To compare CBOW and full softmax on a small vocabulary:

```bash
python train.py --config configs/train.yaml --model.architecture cbow --model.loss softmax --data.max_words 1000 --output cbow.pt
```

Softmax scores every vocabulary word; negative sampling scores one true word and a few sampled words. The trainer builds pairs in memory, so keep `data.max_tokens` bounded for large dumps. Downloads do not overwrite an existing file.

## 2. Explore the vectors

[configs/inference.yaml](configs/inference.yaml) chooses the saved model, query word, and number of neighbors. For nearest words:

```bash
python inference.py --config configs/inference.yaml
```

For an interactive prompt, run `python inference.py --config configs/inference.yaml --interactive true`. Try `similar palermu`, `similarity palermu missina`, `analogy A B C`, or `quit`. A word outside the training vocabulary is reported as OOV.

## 3. Plot a few words

[configs/visualize.yaml](configs/visualize.yaml) chooses seed words, how many neighbors to include, and the PNG filename:

```bash
python visualize.py --config configs/visualize.yaml
```

The plot uses PCA to show the selected word vectors in two dimensions; orange points are seed words, blue points are neighbors. It is a small visual aid, not a full view of the embedding space.

To change the lesson code, start with `src/w2vlab/data.py` (vocabulary and pairs), `src/w2vlab/model.py` (embeddings and losses), and `train.py` (updates).
