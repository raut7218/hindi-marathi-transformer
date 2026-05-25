# Hindi-to-Marathi Transformer Translation Report

## 1. Scope
This submission covers the Part II Transformer-based Hindi-to-Marathi system implemented in this repository: from-scratch encoder pretraining, decoder pretraining, and encoder-decoder fine-tuning for translation.

## 2. System Overview
The model is built from scratch with the following components:
- RoPE for positional encoding
- GQA for attention efficiency
- RMSNorm for stable pre-norm training
- A BERT-like Hindi encoder pretrained with MLM
- A GPT-style Marathi decoder pretrained with CLM
- A warm-started encoder-decoder translation model with cross-attention

## 3. Architecture
### Encoder
- 12 layers
- 768 hidden size
- 12 query heads, 4 KV heads
- FFN size 3072
- Vocab size 45,000
- Parameter count: 110,076,672

### Decoder
- 12 layers
- 768 hidden size
- 12 query heads, 4 KV heads
- FFN size 3584
- Vocab size 50,257
- Parameter count: 123,551,232

### Translation model
The MT model uses the pretrained encoder and decoder, with cross-attention added in the decoder to condition Marathi generation on the Hindi source.

## 4. Data and Tokenization
- Data files: `data/train.hi`, `data/train.mr`, `data/test.hi`, `data/test.mr`
- Tokenization: SentencePiece
- Separate Hindi and Marathi vocabularies
- Max sequence length: 128

## 5. Training Setup
- Optimizer: AdamW
- Schedule: warmup + cosine decay
- Precision: FP16
- Gradient clipping: 1.0
- Batch size: 10 with gradient accumulation of 6
- Effective batch size: 60
- Hardware: Google Colab T4 GPU (16 GB VRAM)

Stages:
- `mlm`: Hindi encoder pretraining
- `clm`: Marathi decoder pretraining
- `mt`: translation fine-tuning
- `eval`: final BLEU/CHRF++ evaluation
- `plot`: metrics plotting

## 6. Evaluation
Metrics required by the assignment are included:
- Train and validation loss
- Train and validation BLEU-100
- Train and validation CHRF++-100

The repository contains the required plots:
- `checkpoints_colab/plots/loss.png`
- `checkpoints_colab/plots/bleu_100.png`
- `checkpoints_colab/plots/chrfpp_100.png`

## 7. Final MT Results
Extracted from `checkpoints_colab/mt/mt_metrics.csv` at step 1000.

| Split | BLEU (raw) | BLEU (%) | CHRF++ |
|---|---:|---:|---:|
| train | 0.0646 | 6.46 | 4.7675 |
| valid | 0.0607 | 6.07 | 4.6458 |

## 8. Key Findings
- Loss decreases across MLM, CLM, and MT, confirming correct training behavior.
- BLEU and CHRF++ remain low because the models are heavily undertrained under the available compute budget.
- The staged freeze/unfreeze strategy is functional, but longer pretraining would likely improve results.

## 9. Artifacts
The repository includes the required deliverables:
- Checkpoints for MLM, CLM, and MT
- Metric logs in CSV and JSONL
- Plot images
- Resolved configs for each stage
- Training, evaluation, and plotting scripts

## 10. Limitations and Future Work
Main limitations:
- Limited training steps
- Greedy decoding only
- No KV cache
- Minimal data filtering

Likely improvements:
- Longer pretraining and fine-tuning
- Beam search decoding
- Better corpus filtering and normalization
- Shared tokenizer experiments

## 11. Reproducibility
Run from the repository root:

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/colab_t4.yaml --stage mlm
python scripts/train.py --config configs/colab_t4.yaml --stage clm
python scripts/train.py --config configs/colab_t4.yaml --stage mt
python scripts/train.py --config configs/colab_t4.yaml --stage eval
python scripts/train.py --config configs/colab_t4.yaml --stage plot
```

## 12. Development Disclosure
- Hardware used: Google Colab T4 GPU
- LLM assistance used: GitHub Copilot Chat and Gemini
- The final code and report were reviewed and edited before submission
