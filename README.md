# Bumblebee — Transformer from Scratch

A from-scratch implementation of the original encoder-decoder Transformer architecture from **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**, created as a university software project at Saarland University in **2022**.

The goal was to understand the Transformer architecture by implementing it with basic PyTorch building blocks rather than using `nn.Transformer` or `nn.MultiheadAttention`.

The model was trained for **German → English machine translation** on the Multi30k dataset.

## Highlights

- Transformer **encoder + decoder** implemented from scratch
- Multi-head self-attention
- Encoder-decoder cross-attention
- Causal and padding masks
- Residual connections and layer normalization
- Feed-forward sublayers
- Token and positional embeddings
- Multiple encoder / decoder layers
- End-to-end training and inference pipeline
- **48-model hyperparameter grid search**
- Trained locally on an **NVIDIA RTX 3070 8 GB**
- Baseline BLEU score: **31.7**
- Final reported BLEU score: **39.4**

## Project Results

The initial baseline model achieved a BLEU score of **31.7**.

I then tuned:

- embedding dimension
- number of attention heads
- learning rate
- number of encoder / decoder layers

After a 48-run grid search and additional manual tuning, the final model used:

| Parameter | Value |
|---|---:|
| `d_model` | 1024 |
| Attention heads | 8 |
| Encoder / decoder layers | 8 |
| Feed-forward dimension | 4096 |
| Learning rate | 0.0001 |
| Dropout | 0.1 |

The final model achieved a reported test loss of **1.51** and BLEU score of **39.4**.

## Generalization

The model translated sentences resembling the Multi30k training distribution reasonably well, but performed poorly on out-of-distribution language such as conversational questions.

For example, sentences describing images translated much better than questions such as:

> *Wie geht es dir?*

This was a useful demonstration of how strongly model behavior depends on the training data distribution.

## Original Report

The full project report contains the architecture description, dataset setup, training procedure, hyperparameter experiments, results, example translations, and discussion:

**[Read the original project report](report/laurent_hug_build_your_own_transformer_Report.pdf)**

## Looking Back

This repository contains the original **2022** implementation and is intentionally kept close to the submitted university project.

Revisiting the code years later, there are a couple of things I would do differently today:

- The attention logits are scaled by `sqrt(d_model)` rather than `sqrt(d_k)`. With multiple attention heads, this makes the attention distribution softer than intended.
- During hyperparameter tuning, a subset of the test set was used for model selection. Today I would keep the test set completely isolated and perform model selection only on validation data.

I have kept these details visible rather than rewriting the historical implementation, since the repository is also a record of how I originally approached the problem.

## Repository Structure

```text
bumblebee/
├── Bumblebee.py
├── BumblebeeTranslatorModel.py
├── Bumblebee_Encoder.py
├── Bumblebee_Decoder_2.py
├── MultiHeadAttention.py
├── Processing.py
├── translate.py
└── report/
    └── laurent_hug_build_your_own_transformer_Report.pdf
```

## Tech

- Python
- PyTorch
- torchtext
- spaCy
- NumPy

---

Built in 2022 as part of a Saarland University software project.
