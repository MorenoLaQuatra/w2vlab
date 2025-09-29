# Word2Vec from Scratch - Template

This is a template for a live coding session for MSc students at UKE.
Most utility code is provided. Core model and data logic must be implemented during the lesson. 

## Usage
- Run `train.py` to start training. 
- Inference code is in `inference.py` and is already implemented.
- All function/class signatures are provided as hints.
- Utility and boilerplate code is already implemented.

## Recap:
- **Skip-gram**: Better for rare words, captures more diverse relationships
- **CBOW**: Faster training, better for frequent words

### Architecture
- **W1 (in_embeddings)**: The input embedding matrix that becomes our final word vectors
- **W2 (out_embeddings)**: Output embeddings used only during training

At this point we choose to only keep `W1` as our word vectors, you can hack to also use `W2` averaging the two matrices (DIY).


> [!NOTE]
> This implementation **prioritizes clarity and educational value** over performance optimization.