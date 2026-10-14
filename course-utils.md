# Live course commands

Run these from the repository root after following the [README](README.md). Replace `my-scn-w2v.pt` with any checkpoint you trained. If you have the course models, their names are `scn-vectors-w2v.pt`, `scn-vectors-fasttext.pt`, `it-vectors-w2v.pt`, and `it-vectors-fasttext.pt`.

```bash
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command shapes
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command 'vocab 10'
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command 'word enna'
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command 'id 0'
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command 'weights enna'
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command 'weights enna 25'
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --command 'word nonexistentword'
```

`summary` names the two tables: W1 maps input IDs to vectors; W2 supplies output/target vectors during training. `shapes` traces a batch through CBOW or skip-gram and shows why softmax scores all `V` words while negative sampling scores one target plus `K` negatives. `word` and `id` inspect both directions of the vocabulary mapping; an unknown word demonstrates OOV. `weights` shows the first 8 dimensions unless you supply a number.

For a live prompt without reloading the model each time:

```bash
python model_info.py --config configs/model_info.yaml --model my-scn-w2v.pt --interactive true
```

The prompt accepts the same commands without `--command`, plus `quit`. To make a quick checkpoint with both W1 and W2, without overwriting your existing model:

```bash
python train.py --config configs/train.yaml --data.max_tokens 20000 --training.epochs 1 --output lesson.pt
python model_info.py --config configs/model_info.yaml --model lesson.pt --command shapes
```

For the costly softmax comparison, train a deliberately small vocabulary and inspect its `[B, V]` score tensor:

```bash
python train.py --config configs/train.yaml --data.max_tokens 20000 --data.max_words 500 --model.loss softmax --training.epochs 1 --output softmax.pt
python model_info.py --config configs/model_info.yaml --model softmax.pt --command shapes
```

Checkpoints saved before this inspector contain W1 only: word lookup still works, but W2 and the training path need a newly trained checkpoint. For nearest neighbors or a 2-D plot, use [inference.py](inference.py) or [visualize.py](visualize.py).

## FastText: what changes?

```bash
python train.py --config configs/train.yaml --model.type fasttext --model.architecture skipgram --data.max_tokens 20000 --training.epochs 3 --output my-scn-fasttext.pt
python model_info.py --config configs/model_info.yaml --model my-scn-fasttext.pt --command shapes
python model_info.py --config configs/model_info.yaml --model my-scn-fasttext.pt --command 'ngrams ennaaaa'
python model_info.py --config configs/model_info.yaml --model my-scn-fasttext.pt --command 'weights ennaaaa'
python inference.py --config configs/inference.yaml --model my-scn-fasttext.pt --word ennaaaa
python visualize.py --config configs/visualize.yaml --model my-scn-fasttext.pt --words ennaaaa --output fasttext-oov.png
```

`ennaaaa` has no whole-word ID but shares some character n-grams with known words. Compare its behavior against `my-scn-w2v.pt`. If using Italian, add `--data.corpus itwiki.txt` to the training command and pick an altered spelling of a known Italian word for the OOV demonstration.
