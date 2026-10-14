# w2vlab

A small PyTorch project for **Elaborazione del Linguaggio Naturale (NLP)** at UKE. Train word2vec or fastText on a text file, inspect the model, and try its word vectors. The code is meant to be changed in class.

## Start

```bash
git clone https://github.com/MorenoLaQuatra/word2vec-from-scratch-template.git w2vlab
cd w2vlab
python -m pip install -e .
```

If you already cloned the repository, run `git pull` and `python -m pip install -e .` to use the current code.

> [!IMPORTANT]
> Run commands from the repository root. Every script requires `--config`; flags after the config path override its YAML values. Give each training run a different `--output` name: training replaces that file if it exists.

## 1. Get some text

[configs/download.yaml](configs/download.yaml) downloads Sicilian Wikipedia. Start with 1,000 articles instead of the full dump:

```bash
python download.py --config configs/download.yaml --articles 1000
```

For Italian instead:

```bash
python download.py --config configs/download.yaml --language it --articles 10000 --output itwiki.txt
```

Already have a UTF-8 text file? Skip downloading and give its path to training with `--data.corpus`. Downloads will not replace an existing file.

## 2. Train your models

[configs/train.yaml](configs/train.yaml) explains the settings. These quick runs use the first 20,000 tokens of `scnwiki.txt` and save separate checkpoints:

```bash
python train.py --config configs/train.yaml --data.max_tokens 20000 --training.epochs 3 --output my-scn-w2v.pt
python train.py --config configs/train.yaml --model.type fasttext --model.architecture skipgram --data.max_tokens 20000 --training.epochs 3 --output my-scn-fasttext.pt
```

To use Italian, add `--data.corpus itwiki.txt` and choose an Italian output name. Edit the YAML to change the context window, vector size, vocabulary limit, or number of epochs. `model.loss: negative` uses a few sampled words; `softmax` scores the whole vocabulary. For a small softmax experiment, add `--model.loss softmax --data.max_words 500`.

For the full Italian dump, use `-1` and start with one epoch:

```bash
python train.py --config configs/train.yaml --data.corpus itwiki.txt --data.max_tokens -1 --model.type fasttext --training.epochs 1 --output it-fasttext-full.pt
```

> [!NOTE]
> Training reads the corpus in passes: one for vocabulary counts, then one per epoch. Context windows stop at article boundaries. RAM holds a counting chunk, the vocabulary/subword tables, and a small training buffer—not the full corpus. Exact word counting uses a temporary SQLite file and needs free disk space. A full Wikipedia run can still take many hours.

Training prints its device (`cuda`, `mps`, or `cpu`), vocabulary coverage, model shapes, loss, speed, and ETA. [configs/train.yaml](configs/train.yaml) controls how often progress appears. It saves to `it-fasttext-full.checkpoint.pt` every 10 million examples and after each epoch. To continue, rerun the same command with `--training.resume it-fasttext-full.checkpoint.pt`. Keep the corpus and training settings unchanged; you may increase `--training.epochs`. Resume skips earlier examples without retraining them, but must reread that part of the file. You can also inspect a checkpoint while training continues; the finished `.pt` is smaller.

If many tokens become `<unk>`, raise `data.max_words` for a **new** run; changing it prevents resuming an existing checkpoint.

## 3. Look inside and try the model

[configs/model_info.yaml](configs/model_info.yaml) and [configs/inference.yaml](configs/inference.yaml) set the defaults. Use `--model` to open any checkpoint you trained; replace `my-scn-fasttext.pt` with `my-scn-w2v.pt` to compare:

```bash
python model_info.py --config configs/model_info.yaml --model my-scn-fasttext.pt --interactive true
python inference.py --config configs/inference.yaml --model my-scn-fasttext.pt --interactive true
```

To inspect an unfinished run, use its `.checkpoint.pt` file in either command, for example `--model it-fasttext-full.checkpoint.pt`.

At the **model-info** prompt, try `summary`, `shapes`, `vocab 10`, `word enna`, `id 0`, `weights enna`, and `quit`. FastText also has `ngrams ennaaaa`: an unseen word can get a vector from character pieces it shares with known words. At the **inference** prompt, try `similar enna`, `similarity enna palermu`, `analogy A B C`, and `quit`.

## 4. Plot a few words

[configs/visualize.yaml](configs/visualize.yaml) makes a simple 2-D PCA plot:

```bash
python visualize.py --config configs/visualize.yaml --model my-scn-fasttext.pt --words enna --output my-scn-plot.png
```

To change the lesson code, start with [data.py](src/w2vlab/data.py) (words and pairs), [model.py](src/w2vlab/model.py) (embeddings and losses), and [train.py](train.py) (training loop). More live examples are in [course-utils.md](course-utils.md).
