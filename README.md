# AdiVaani Hindi-Marathi Part II Transformer Pipeline

This repository contains the Part II implementation for the MISN Lab / AdiVaani hiring assignment: from-scratch Hindi encoder MLM pretraining, Marathi GPT-style CLM pretraining, and warm-started Hindi-to-Marathi translation fine-tuning.

The current codebase is centered on a single strict Part II config, [configs/part2_t4.yaml](configs/part2_t4.yaml), which drives the MLM, CLM, MT, eval, and plot stages.

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

## Part II Commands

Train the Hindi BERT-like encoder from scratch:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage mlm
```

Train the Marathi GPT-style decoder-only model from scratch:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage clm
```

Set the warm-start checkpoints in `configs/part2_t4.yaml`:

```yaml
mt:
  encoder_checkpoint: checkpoints_part2/mlm/encoder_mlm_final_step200.pt
  decoder_checkpoint: checkpoints_part2/clm/decoder_clm_final_step200.pt
  freeze_pretrained: false
```

These checkpoints are loaded automatically by the `mt` stage so the translation model starts from the pretrained encoder and decoder rather than random weights.

Fine-tune the encoder-decoder MT model:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage mt
```

Evaluate a saved MT checkpoint by setting `evaluation.checkpoint`, then running:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage eval
```

Generate the required plots:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage plot
```

Plots are written to `checkpoints_part2/plots/`:

- `loss.png`
- `bleu_100.png`
- `chrfpp_100.png`

## Parameter Targets

The strict Part II config uses separate tokenizers:

- Hindi encoder: vocab 45,000, 12 layers, hidden 768, 12 query heads, 4 KV heads, FFN 3072, about 110.08M parameters.
- Marathi decoder: vocab 50,257, 12 layers, hidden 768, 12 query heads, 4 KV heads, FFN 3584, about 123.55M parameters.

## Smoke Tests

```bash
pytest
```
