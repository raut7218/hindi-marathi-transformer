# Hindi to Marathi Transformer MT

This repo trains a small Transformer encoder-decoder with RoPE, GQA, and RMSNorm on the provided Hindi-Marathi data.

## Quick start (Colab)

1. Install deps:

```
pip install -r requirements.txt
```

2. Run tokenizer training and a stage:

```
python scripts/train.py --config configs/fast_t4.yaml --stage mlm
python scripts/train.py --config configs/fast_t4.yaml --stage clm
python scripts/train.py --config configs/fast_t4.yaml --stage mt
```

Checkpoints are saved under `checkpoints/` by default.
