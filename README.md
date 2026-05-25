# AdiVaani Hindi-Marathi Part II Transformer Pipeline

This repository contains the Part II implementation for the MISN Lab / AdiVaani hiring assignment: from-scratch Hindi encoder MLM pretraining, Marathi GPT-style CLM pretraining, and warm-started Hindi-to-Marathi translation fine-tuning.

## Architecture

The Part II models use the required architectural modifications throughout:

- **Rotary Positional Embeddings (RoPE)** instead of sinusoidal or learned positional embeddings (`src/models/rope.py`)
- **Grouped Query Attention (GQA)** with 12 query heads and 4 KV heads (`src/models/attention.py`)
- **RMSNorm** instead of LayerNorm (`src/models/norms.py`)
- **BERT-like encoder** pretrained with MLM only (no NSP) for Hindi understanding
- **GPT-2-style decoder** pretrained with CLM for Marathi generation
- **Encoder-decoder with cross-attention** for Hindi-to-Marathi translation

### Parameter Counts

| Component | Parameters |
|---|---:|
| Hindi encoder (BERT-like, ~110M target) | 110,076,672 |
| Marathi decoder (GPT-2-like, ~124M target) | 123,551,232 |

## Repository Structure

| Path | Purpose |
|---|---|
| `src/models/transformer.py` | Encoder, decoder, MT decoder, Seq2Seq model |
| `src/models/attention.py` | Grouped Query Attention with RoPE |
| `src/models/rope.py` | Rotary positional embedding cache |
| `src/models/norms.py` | RMSNorm |
| `src/data/tokenizer.py` | SentencePiece tokenizer wrapper |
| `src/data/datasets.py` | Monolingual + parallel datasets, MLM/CLM/MT batch builders |
| `src/utils/metrics.py` | BLEU and ChrF++ via sacrebleu |
| `src/utils/decoding.py` | Greedy decoding |
| `src/utils/schedule.py` | Warmup + cosine LR schedule |
| `src/utils/checkpoint.py` | Atomic checkpoint save/load |
| `src/utils/distributed.py` | DDP utilities |
| `scripts/train.py` | Main CLI for MLM, CLM, MT, eval, plot |
| `configs/colab_t4.yaml` | Single Colab T4 config |
| `configs/part2_t4.yaml` | Multi-GPU DDP config |
| `tests/test_part2.py` | Unit tests for shapes, param counts, and plots |
| `Report.md` | Detailed technical report |

## Deliverables Included in Repo

The following training artifacts are tracked in Git:

- **Plots**: `checkpoints_colab/plots/loss.png`, `bleu_100.png`, `chrfpp_100.png`
- **Metrics**: CSV and JSONL logs for all 3 stages under `checkpoints_colab/{mlm,clm,mt}/`
- **Resolved configs**: Per-stage resolved YAML in each checkpoint directory
- **Tokenizer models**: `data/spm_hi_45k.model`, `data/spm_mr_50257.model`

### Checkpoint Files (Not in Git)

Model checkpoints are too large for Git. After training, they are saved at:

- `checkpoints_colab/mlm/encoder_mlm_final_step1000.pt` (~440 MB)
- `checkpoints_colab/clm/decoder_clm_final_step1000.pt` (~494 MB)
- `checkpoints_colab/mt/mt_final_step1000.pt` (~1010 MB)

To reproduce, run the training stages below. The checkpoints will be regenerated.

## Install

```bash
pip install -r requirements.txt
```

## Data Setup

Place the Hindi-Marathi parallel corpus in the `data/` directory:
- `data/train.hi` — Hindi training sentences
- `data/train.mr` — Marathi training sentences (aligned)
- `data/test.hi` — Hindi test sentences
- `data/test.mr` — Marathi test sentences (aligned)

## Training (Google Colab Single T4)

```bash
cd /content/hindi-marathi-transformer
pip install -r requirements.txt

# Stage 1: Pretrain Hindi encoder with MLM
python scripts/train.py --config configs/colab_t4.yaml --stage mlm

# Stage 2: Pretrain Marathi decoder with CLM
python scripts/train.py --config configs/colab_t4.yaml --stage clm

# Stage 3: Fine-tune encoder-decoder for translation
python scripts/train.py --config configs/colab_t4.yaml --stage mt
```

Or use the launcher script:

```bash
bash scripts/launch_colab.sh mlm
bash scripts/launch_colab.sh clm
bash scripts/launch_colab.sh mt
```

## Multi-GPU Training

```bash
GPUS=2 bash scripts/launch_distributed.sh train mlm
GPUS=2 bash scripts/launch_distributed.sh train clm
GPUS=2 bash scripts/launch_distributed.sh train mt
```

## Evaluation and Plotting

```bash
# Evaluate MT checkpoint on test set
python scripts/train.py --config configs/colab_t4.yaml --stage eval

# Generate loss, BLEU, and ChrF++ plots
python scripts/train.py --config configs/colab_t4.yaml --stage plot
```

## Smoke Tests

```bash
pytest
```

Tests verify:
- Encoder parameter count = 110,076,672
- Decoder parameter count = 123,551,232
- Output shapes for MLM, CLM, and MT forward passes
- Plot generation from metric files

## Training Configuration

The Colab T4 config uses:

```yaml
batch_size: 10
grad_accum_steps: 6        # effective batch = 60
max_steps: 1000             # per stage
warmup_steps: 100
lr: 3.0e-4                  # MLM/CLM base LR
stage_lrs.mt: 1.0e-4        # MT fine-tuning LR
label_smoothing.mt: 0.1     # label smoothing for MT only
amp_dtype: fp16
cross_only_steps: 300       # freeze pretrained weights for first 300 MT steps
pretrained_lr_mult: 0.3     # 0.3x LR for pretrained params during MT
```

## GPU Hardware

- **Primary**: Google Colab T4 (16 GB VRAM, FP16)
- **Supported**: Multi-GPU via DDP (NCCL backend)

## LLM Assistance Disclosure

GitHub Copilot Chat and Gemini were used during development for code assistance, debugging, and report editing. All code and decisions were reviewed by the author. See Report.md Section 34 for details.
