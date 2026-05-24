# AdiVaani Hindi-Marathi Part II Transformer Pipeline

This repository contains the Part II implementation for the MISN Lab / AdiVaani hiring assignment: from-scratch Hindi encoder MLM pretraining, Marathi GPT-style CLM pretraining, and warm-started Hindi-to-Marathi translation fine-tuning.

The codebase now has two ready-to-run configs:

- `configs/colab_t4.yaml`: single Google Colab T4, model-only checkpoints, no DDP.
- `configs/part2_t4.yaml`: explicit multi-GPU/Kaggle style training.

The Part II models use the required architectural changes throughout:

- Rotary positional embeddings instead of sinusoidal or learned positional embeddings
- Grouped Query Attention
- RMSNorm
- BERT-like encoder pretraining with MLM only, no NSP
- GPT-2-style decoder-only CLM pretraining with no cross-attention

## Install

```bash
pip install -r requirements.txt
```

`numpy` is pinned below 2 because some current Torch builds emit runtime warnings or fail with NumPy 2.x.

## Google Colab Single T4

Use the Colab launcher for a single T4. It runs plain Python, avoids NCCL/DDP, and writes smaller model-only checkpoints under `checkpoints_colab/`.

```bash
cd /content/hindi-marathi-transformer
pip install -r requirements.txt

bash scripts/launch_colab.sh train mlm
bash scripts/launch_colab.sh train clm
bash scripts/launch_colab.sh train mt
```

Equivalent direct commands:

```bash
python scripts/train.py --config configs/colab_t4.yaml --stage mlm
python scripts/train.py --config configs/colab_t4.yaml --stage clm
python scripts/train.py --config configs/colab_t4.yaml --stage mt
```

Evaluate or plot after training:

```bash
python scripts/train.py --config configs/colab_t4.yaml --stage eval
python scripts/train.py --config configs/colab_t4.yaml --stage plot
```

For evaluation, set `evaluation.checkpoint` in `configs/colab_t4.yaml` to a saved MT checkpoint path first.

## Kaggle Or Multi-GPU

Multi-GPU training is still supported, but request it explicitly:

```bash
GPUS=2 bash scripts/launch_distributed.sh train mlm
GPUS=2 bash scripts/launch_distributed.sh train clm
GPUS=2 bash scripts/launch_distributed.sh train mt
```

Single-GPU through the same launcher:

```bash
GPUS=1 bash scripts/launch_distributed.sh train mt
```

## Checkpointing

Checkpoint writes are atomic: the code writes `*.tmp` first and then replaces the final file. This prevents corrupt partial checkpoints after interrupted writes.

The Colab config uses:

```yaml
training:
  save_every: 0
  save_optimizer_state: false
  save_scaler_state: false
  output_dir: checkpoints_colab
```

That disables large intermediate checkpoints and keeps final checkpoints smaller. Multi-GPU saves synchronize ranks before and after rank 0 writes, so other ranks do not continue into DDP collectives while rank 0 is checkpointing.

## Parameter Targets

The strict Part II configs use separate tokenizers:

- Hindi encoder: vocab 45,000, 12 layers, hidden 768, 12 query heads, 4 KV heads, FFN 3072, about 110.08M parameters.
- Marathi decoder: vocab 50,257, 12 layers, hidden 768, 12 query heads, 4 KV heads, FFN 3584, about 123.55M parameters.

## Smoke Tests

```bash
pytest
```
