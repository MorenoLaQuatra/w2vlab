# w2vlab

A small PyTorch word2vec and fastText lab for the course Elaborazione del Linguaggio Naturale (NLP) at UKE. It creates word/context pairs, trains with skip-gram or CBOW, and lets you compare negative sampling with full softmax. The fastText option adds character n-grams to the input word vectors.

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

[configs/train.yaml](configs/train.yaml) reads `scnwiki.txt`, uses the first 200,000 tokens (`data.max_tokens`), and trains CBOW with negative sampling for 20 epochs. It saves `vectors.pt`:

```bash
python train.py --config configs/train.yaml
```

Training prints an example pair, tensor shapes, progress, and loss. Rare words share `<unk>`. To compare CBOW and full softmax on a small vocabulary:

```bash
python train.py --config configs/train.yaml --model.architecture cbow --model.loss softmax --data.max_words 1000 --output cbow.pt
```

Softmax scores every vocabulary word; negative sampling scores one true word and a few sampled words. The trainer builds pairs in memory, so keep `data.max_tokens` bounded for large dumps. Downloads do not overwrite an existing file.

For the next lesson, train the fastText variant with the same script. This quick run uses Sicilian; add `--data.corpus itwiki.txt` when your Italian corpus is ready:

```bash
python train.py --config configs/train.yaml --model.type fasttext --model.architecture skipgram --data.max_tokens 20000 --training.epochs 3 --output fasttext.pt
```

The [fastText subword model](https://aclanthology.org/Q17-1010/) sums a whole-word vector and character n-gram vectors. This lab keeps an explicit n-gram vocabulary instead of hashing, so you can inspect it. An unseen word gets a vector only when it shares learned n-grams; W2 and the training loss are unchanged.

## 2. Explore the vectors

[configs/inference.yaml](configs/inference.yaml) chooses the saved model, query word, and number of neighbors. For nearest words:

```bash
python inference.py --config configs/inference.yaml
```

For an interactive prompt, run `python inference.py --config configs/inference.yaml --interactive true`. Try `similar palermu`, `similarity palermu missina`, `analogy A B C`, or `quit`. Word2vec reports an unseen word as OOV; fastText can build its vector from shared n-grams. Try `python inference.py --config configs/inference.yaml --model fasttext.pt --word ennaaaa`.

## 3. Inspect the model in class

[configs/model_info.yaml](configs/model_info.yaml) chooses the checkpoint. These commands show the model tables, the shape of each step, and the word/ID mapping:

```bash
python model_info.py --config configs/model_info.yaml
python model_info.py --config configs/model_info.yaml --command shapes
python model_info.py --config configs/model_info.yaml --command 'word enna'
```

See [course-utils.md](course-utils.md) for more copy-paste commands and the interactive mode. New checkpoints include both W1 and W2; older `vectors.pt` files still work for word lookup but need retraining to show W2 and the training path.

With `fasttext.pt`, try `python model_info.py --config configs/model_info.yaml --model fasttext.pt --command 'ngrams ennaaaa'` to see the pieces of an OOV word.

## 4. Plot a few words

[configs/visualize.yaml](configs/visualize.yaml) chooses seed words, how many neighbors to include, and the PNG filename:

```bash
python visualize.py --config configs/visualize.yaml
```

The plot uses PCA to show the selected word vectors in two dimensions; orange points are seed words, blue points are neighbors. It is a small visual aid, not a full view of the embedding space.

To change the lesson code, start with `src/w2vlab/data.py` (vocabulary and pairs), `src/w2vlab/model.py` (embeddings and losses), and `train.py` (updates).
