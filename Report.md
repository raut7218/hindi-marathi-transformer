# Hindi-to-Marathi Transformer Translation System Report

## 1. Executive Summary

This project implements a from-scratch Transformer training pipeline for Hindi-to-Marathi neural machine translation. The solution is designed around three stages:

1. Hindi encoder pretraining with Masked Language Modeling (MLM).
2. Marathi decoder pretraining with Causal Language Modeling (CLM).
3. Warm-started encoder-decoder fine-tuning for Hindi-to-Marathi machine translation.

The repository does not depend on Hugging Face model classes for the model architecture. The Transformer components are implemented directly in `src/models`, including:

- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm
- BERT-like bidirectional encoder
- GPT-style autoregressive decoder
- Encoder-decoder model with cross-attention for translation

The strict Part II configuration targets the required model sizes:

| Component | Configuration | Parameter Count |
|---|---:|---:|
| Hindi encoder | 12 layers, 768 hidden, 12 query heads, 4 KV heads, FFN 3072, vocab 45000 | 110,076,672 |
| Marathi decoder | 12 layers, 768 hidden, 12 query heads, 4 KV heads, FFN 3584, vocab 50257 | 123,551,232 |

The implementation is intentionally modular: data loading, tokenization, model blocks, metrics, checkpointing, learning-rate schedule, and training orchestration are separated into small files. This makes it easier to defend in an interview because each design decision maps to a clear part of the codebase.

## 2. Problem Statement

The assignment asks for a modern Transformer-based system for Hindi-to-Marathi translation. The core challenge is not only to build a translation model, but to show understanding of how pretrained language representations can be combined into a sequence-to-sequence system.

The solution follows this idea:

- Hindi source sentences should be encoded with a bidirectional encoder because understanding the full source sentence is important before translation.
- Marathi target sentences should be generated with an autoregressive decoder because translation output is produced token by token.
- Pretraining the encoder and decoder separately gives each side a useful starting point before translation fine-tuning.
- Cross-attention connects the pretrained encoder and decoder during the translation stage.

In simple interview language:

> I built a mini BERT-style encoder for Hindi understanding, a mini GPT-style decoder for Marathi generation, and then connected them through cross-attention to fine-tune a full Hindi-to-Marathi translation model.

## 3. Repository Structure

| Path | Purpose |
|---|---|
| `scripts/train.py` | Main command-line entrypoint for MLM, CLM, MT, evaluation, and plotting |
| `src/models/transformer.py` | Encoder, decoder, MT decoder, seq2seq model, masks, feed-forward blocks |
| `src/models/attention.py` | Grouped Query Attention implementation with RoPE support |
| `src/models/rope.py` | Rotary embedding cache and rotation function |
| `src/models/norms.py` | RMSNorm implementation |
| `src/data/tokenizer.py` | SentencePiece tokenizer wrapper and tokenizer training |
| `src/data/datasets.py` | Monolingual and parallel datasets plus MLM, CLM, MT batch builders |
| `src/utils/schedule.py` | Warmup plus cosine learning-rate schedule |
| `src/utils/metrics.py` | BLEU and chrF++ computation using sacrebleu |
| `src/utils/decoding.py` | Greedy decoding for evaluation |
| `src/utils/checkpoint.py` | Atomic save/load checkpoint helpers |
| `src/utils/params.py` | Parameter counting helpers |
| `configs/part2_t4.yaml` | Main strict Part II multi-GPU training configuration |
| `configs/colab_t4.yaml` | Single-GPU Google Colab T4 configuration |
| `tests/test_part2.py` | Shape tests, parameter count tests, and plot-generation tests |

The codebase is organized around strict Part II configurations for both Colab and Kaggle-style runs. The `mt` section stores the encoder and decoder warm-start checkpoints used by the translation stage, while `freeze_pretrained`, `cross_only_steps`, and `pretrained_lr_mult` control how pretrained weights are used during MT fine-tuning.

## 4. High-Level Architecture

The complete system has three model forms.

### 4.1 Encoder-Only Model

The encoder is implemented by `EncoderModel`.

It is used for Hindi MLM pretraining. It receives a Hindi sentence where some tokens are masked, processes the whole sentence bidirectionally, and predicts only the masked tokens.

Important properties:

- No causal mask, so every token can attend to every non-padding token.
- Uses GQA self-attention.
- Uses RoPE for positional information.
- Uses RMSNorm before attention and feed-forward layers.
- Uses tied input/output embeddings for the MLM head.

The encoder answers:

> What does this Hindi sentence mean when all surrounding context is visible?

### 4.2 Decoder-Only Model

The decoder is implemented by `DecoderModel`.

It is used for Marathi CLM pretraining. It predicts the next Marathi token from previous tokens.

Important properties:

- Uses a causal mask, so token `t` cannot attend to tokens after `t`.
- Uses GQA self-attention.
- Uses RoPE for positional information.
- Uses RMSNorm.
- Uses tied input/output embeddings.

The decoder answers:

> Given previous Marathi tokens, what Marathi token should come next?

### 4.3 Encoder-Decoder Translation Model

The translation model is implemented by `Seq2SeqModel`.

It combines:

- `EncoderModel` for the Hindi source sentence.
- `MTDecoderModel` for Marathi generation.

The MT decoder is similar to the CLM decoder, but each block adds a cross-attention sublayer:

1. Causal self-attention over previously generated Marathi tokens.
2. Cross-attention from Marathi decoder states to Hindi encoder states.
3. Feed-forward network.

The translation model answers:

> Given the full Hindi source sentence and the Marathi prefix generated so far, what Marathi token should come next?

## 5. Transformer Block Design

The code uses a pre-norm residual Transformer design.

### 5.1 Encoder Block

Each encoder block does:

1. RMSNorm on the input.
2. GQA self-attention.
3. Residual connection.
4. RMSNorm.
5. Feed-forward network with GELU.
6. Residual connection.

Formula-style:

```text
x = x + SelfAttention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Why pre-norm?

- Pre-norm Transformers are usually more stable during training.
- Gradients flow more directly through residual paths.
- This matters because the models are deep enough, 12 layers in the strict config, to suffer from instability if normalization is placed poorly.

### 5.2 GPT Decoder Block

Each CLM decoder block does:

```text
x = x + CausalSelfAttention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

The causal mask makes sure the decoder cannot cheat by reading future tokens.

### 5.3 MT Decoder Block

Each MT decoder block does:

```text
x = x + CausalSelfAttention(RMSNorm(x))
x = x + CrossAttention(RMSNorm(x), encoder_output)
x = x + FFN(RMSNorm(x))
```

The cross-attention layer is the main bridge between Hindi understanding and Marathi generation.

Interview explanation:

> Self-attention tells the decoder what it has already generated. Cross-attention tells it what source sentence it is translating.

## 6. Grouped Query Attention

GQA is implemented in `src/models/attention.py`.

Standard multi-head attention gives every query head its own key head and value head. In GQA, there are more query heads than key/value heads. In our strict config:

```text
n_heads = 12
n_kv_heads = 4
```

So every KV head is shared by 3 query heads.

### 6.1 Why GQA?

GQA gives a middle ground between Multi-Head Attention and Multi-Query Attention.

| Attention Type | Query Heads | KV Heads | Trade-off |
|---|---:|---:|---|
| MHA | Many | Same as query heads | Highest flexibility, more memory |
| MQA | Many | 1 | Lowest KV memory, possible quality loss |
| GQA | Many | Few groups | Good efficiency with less quality loss |

In autoregressive generation, keys and values are expensive because they are repeatedly used as the sequence grows. GQA reduces key/value projection size and can reduce memory bandwidth pressure.

### 6.2 How It Works in Code

The attention module creates:

- `q_proj`: projects to `n_heads * head_dim`
- `k_proj`: projects to `n_kv_heads * head_dim`
- `v_proj`: projects to `n_kv_heads * head_dim`
- `out_proj`: projects back to `d_model`

If PyTorch supports it, the code uses:

```python
F.scaled_dot_product_attention(..., enable_gqa=True)
```

If not, it falls back to manual attention and repeats KV heads with `repeat_interleave`.

Interview answer:

> I used GQA because it keeps many query heads for expressiveness but shares fewer key/value heads for efficiency. With 12 query heads and 4 KV heads, each KV head serves 3 query heads.

## 7. Rotary Positional Embeddings

RoPE is implemented in `src/models/rope.py` and applied inside `GQAAttention`.

Traditional Transformers need positional information because attention itself is permutation-invariant. Instead of adding learned or sinusoidal position embeddings to token embeddings, RoPE rotates query and key vectors according to token position.

### 7.1 Why RoPE?

RoPE was selected because:

- It injects position directly into attention scores.
- It gives the model relative-position awareness.
- It does not add learned positional embedding parameters.
- It is a common modern Transformer choice.

### 7.2 How It Works in Code

The head dimension is split into pairs:

```text
(x0, x1), (x2, x3), ...
```

Each pair is rotated using cached cosine and sine values:

```text
rot_x1 = x1 * cos - x2 * sin
rot_x2 = x1 * sin + x2 * cos
```

The code applies RoPE to:

- Queries
- Keys

It does not apply RoPE to values.

Why not values?

> RoPE affects attention score computation through Q and K. Values are the content being mixed, so rotating V is unnecessary.

### 7.3 Cross-Attention Detail

In the MT decoder, cross-attention is created with `use_rope=False`.

Reason:

- Self-attention operates within one sequence, so token positions are directly comparable.
- Cross-attention compares target-side queries to source-side keys from a different sequence.
- Applying one shared RoPE scheme across source and target positions can be less clean because source position 5 and target position 5 are not the same semantic timeline.

Interview answer:

> I use RoPE in encoder self-attention and decoder self-attention. I disable it in cross-attention because cross-attention connects two different sequences, source and target, where absolute positions are not directly aligned.

## 8. RMSNorm

RMSNorm is implemented in `src/models/norms.py`.

It normalizes by the root mean square of the hidden dimension:

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

Unlike LayerNorm, it does not subtract the mean.

### 8.1 Why RMSNorm?

RMSNorm was selected because:

- It is simpler than LayerNorm.
- It reduces computation slightly.
- It is widely used in modern LLM-style architectures.
- It works well with pre-norm residual Transformer blocks.

Interview answer:

> RMSNorm keeps the important rescaling behavior of normalization while avoiding mean-centering. It is a modern, efficient normalization choice and helps stabilize deep Transformer training.

## 9. Feed-Forward Network

The feed-forward block is implemented as:

```text
Linear(d_model -> ffn_dim)
GELU
Linear(ffn_dim -> d_model)
Dropout
```

Both linear layers use `bias=False`, which matches the minimal modern Transformer style used elsewhere in this repo.

Strict config values:

| Model | d_model | ffn_dim |
|---|---:|---:|
| Encoder | 768 | 3072 |
| Decoder | 768 | 3584 |

The decoder FFN is slightly larger than the common 4x ratio to reach the required approximately 124M parameter target with the selected vocabulary and tied embeddings.

## 10. Tokenization Strategy

Tokenization is handled by SentencePiece in `src/data/tokenizer.py`.

The strict config uses separate tokenizers:

```yaml
tokenizers:
  shared: false
  src:
    vocab_size: 45000
  tgt:
    vocab_size: 50257
```

### 10.1 Why SentencePiece?

SentencePiece is useful for Indic languages because:

- It works directly on Unicode text.
- It does not require whitespace-based pre-tokenization.
- It learns subword units, which helps with morphologically rich languages.
- It supports a custom `[MASK]` token required for MLM.

### 10.2 Why Separate Tokenizers?

Separate tokenizers let the Hindi encoder and Marathi decoder have vocabularies sized around their own parameter targets:

- Hindi encoder vocab: 45,000
- Marathi decoder vocab: 50,257

This also mirrors the architecture: the source language is encoded and the target language is generated.

A shared tokenizer would be a reasonable alternative because Hindi and Marathi both use Devanagari and share many subword patterns. However, separate tokenizers make parameter control easier and give target-side generation its own vocabulary.

Interview answer:

> I used SentencePiece because it is robust for raw Unicode and subword modeling. I used separate tokenizers in the final config mainly to hit the required encoder and decoder parameter budgets cleanly and to let Hindi and Marathi vocabularies specialize.

## 11. Data Pipeline

The data lives in:

- `data/train.hi`
- `data/train.mr`
- `data/test.hi`
- `data/test.mr`

### 11.1 MonolingualDataset

Used for MLM and CLM.

It:

- Reads non-empty lines from a single file.
- Encodes each sentence with BOS and EOS.
- Truncates to `max_seq_len`.

For MLM:

- Hindi file is used: `train.hi`

For CLM:

- Marathi file is used: `train.mr`

### 11.2 ParallelDataset

Used for machine translation.

It:

- Reads aligned Hindi and Marathi files.
- Checks both sides have the same number of lines.
- Encodes source with source tokenizer.
- Encodes target with target tokenizer.
- Returns `(src_ids, tgt_ids)` pairs.

### 11.3 Padding and Attention Masks

`pad_sequences` pads variable-length examples to the longest sequence in the batch.

The code creates attention masks where:

```text
1 = real token
0 = padding token
```

Then model-side helper `make_pad_mask` converts that to additive attention masking:

```text
padding positions get a large negative value
```

That large negative value makes softmax assign near-zero probability to padding tokens.

## 12. MLM Pretraining

MLM batching is implemented in `make_mlm_batch`.

The encoder is trained to predict randomly masked Hindi tokens.

### 12.1 Masking Rule

For each non-special token:

- 15 percent probability of selecting the token for prediction.
- If selected:
  - 80 percent replaced with `[MASK]`
  - 10 percent replaced with a random token
  - 10 percent kept unchanged
- If not selected:
  - label becomes `-100`, so loss ignores it

Special tokens are never predicted:

- PAD
- BOS
- EOS

### 12.2 Loss

The model outputs logits of shape:

```text
batch_size x sequence_length x vocab_size
```

Cross-entropy is computed over only masked positions:

```python
F.cross_entropy(..., ignore_index=-100)
```

### 12.3 Why MLM for Encoder?

The encoder should learn bidirectional understanding. MLM is appropriate because the model uses left and right context to reconstruct masked tokens.

Interview answer:

> MLM trains the encoder to build contextual Hindi representations. Since translation requires understanding the whole source sentence, a bidirectional objective is better than left-to-right prediction for the source side.

## 13. CLM Pretraining

CLM batching is implemented in `make_clm_batch`.

The decoder is trained on Marathi text to predict the next token.

### 13.1 Label Shift

Input:

```text
BOS token1 token2 token3
```

Labels:

```text
token1 token2 token3 EOS
```

In code:

```python
labels[:, :-1] = input_ids[:, 1:]
labels[:, -1] = -100
```

Padding positions are also ignored.

### 13.2 Causal Mask

The decoder creates a causal mask in `make_causal_mask`.

This prevents token `i` from attending to tokens after `i`.

### 13.3 Why CLM for Decoder?

Translation generation is autoregressive. At inference time, the model produces one Marathi token at a time. CLM pretraining matches that generation behavior.

Interview answer:

> CLM teaches the decoder fluent Marathi generation. During translation fine-tuning, the decoder already knows how Marathi sequences should continue, and cross-attention teaches it how to condition that generation on Hindi input.

## 14. Machine Translation Fine-Tuning

MT batching is implemented in `make_mt_batch`.

The translation model receives:

- Hindi source ids
- Hindi source mask
- Marathi decoder input ids
- Marathi target mask
- Marathi labels

### 14.1 Teacher Forcing

During training, the decoder input is the gold Marathi target shifted right:

```text
decoder input: BOS y1 y2 y3
labels:        y1  y2 y3 EOS
```

This is teacher forcing. It makes training efficient because all target positions can be trained in parallel.

### 14.2 Cross-Attention

The encoder first produces:

```text
enc_out = encoder(src_ids, src_mask)
```

Then each MT decoder block attends to `enc_out` using cross-attention.

In cross-attention:

- Queries come from decoder hidden states.
- Keys and values come from encoder outputs.

Interview answer:

> In self-attention, Q, K, and V come from the same sequence. In cross-attention, Q comes from the target decoder state, while K and V come from the source encoder output.

## 15. Warm Start Strategy

Warm-starting is implemented in `load_warm_start`.

The MT model can load:

- A pretrained encoder checkpoint into `model.encoder`.
- A pretrained decoder checkpoint into `model.decoder`.

The decoder load uses `strict=False` because:

- The CLM decoder does not have cross-attention layers.
- The MT decoder adds cross-attention layers.
- Therefore, pretrained self-attention, FFN, embeddings, and norm weights can load, but cross-attention weights are newly initialized.

Interview answer:

> I use strict loading for the encoder because the architecture matches. For the decoder, I use non-strict loading because the MT decoder has additional cross-attention layers that were not present during CLM pretraining.

### 15.1 Optional Freezing

The config supports:

```yaml
mt:
  freeze_pretrained: false
```

If set to true, all non-cross-attention parameters are frozen.

Why is this useful?

- It can isolate learning to the new cross-attention bridge.
- It reduces memory and compute.
- It can be useful for debugging whether cross-attention learns alignment.

Why default false?

- Full fine-tuning usually gives better final translation quality when enough data and compute are available.

## 16. Training Orchestration

All stages are controlled by `scripts/train.py`.

### 16.1 Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run smoke tests:

```bash
pytest
```

Train Hindi MLM encoder:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage mlm
```

Train Marathi CLM decoder:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage clm
```

Fine-tune machine translation:

```bash
python scripts/train.py --config configs/part2_t4.yaml --stage mt
```

For Google Colab single-T4 runs, use the Colab config and launcher:

```bash
bash scripts/launch_colab.sh train mlm
bash scripts/launch_colab.sh train clm
bash scripts/launch_colab.sh train mt
```

For Kaggle or explicit multi-GPU runs:

```bash
GPUS=2 bash scripts/launch_distributed.sh train mlm
GPUS=2 bash scripts/launch_distributed.sh train clm
GPUS=2 bash scripts/launch_distributed.sh train mt
```

Evaluate a checkpoint:

```bash
python scripts/train.py --config configs/colab_t4.yaml --stage eval
```

Generate plots:

```bash
python scripts/train.py --config configs/colab_t4.yaml --stage plot
```

### 16.2 Training Loop

Each training stage follows the same pattern:

1. Select device.
2. Select AMP dtype.
3. Prepare output directory.
4. Save resolved config.
5. Prepare tokenizer or tokenizers.
6. Build dataset.
7. Build model.
8. Optionally load warm-start weights.
9. Build optimizer and gradient scaler.
10. Optionally auto-tune batch size.
11. Train for `max_steps`.
12. Log metrics.
13. Periodically evaluate.
14. Periodically save checkpoint when enabled.
15. Save final checkpoint.

Checkpoint writes are atomic: the code saves to a temporary file and then replaces the final path. In multi-GPU mode, ranks synchronize around checkpoint saves so non-main ranks do not continue into DDP collectives while rank 0 is writing. The Colab config disables optimizer-state checkpoints by default to reduce file size and avoid notebook storage write failures.

### 16.3 Optimizer

The optimizer is AdamW.

Why AdamW?

- It is a standard optimizer for Transformers.
- Decoupled weight decay is usually better than L2 regularization inside Adam.
- It handles sparse and noisy gradient behavior better than vanilla SGD for this type of model.

On CUDA, the code tries to use fused AdamW when available.

### 16.4 Learning-Rate Schedule

The schedule is warmup plus cosine decay:

1. During warmup, LR grows linearly from 0 to base LR.
2. After warmup, LR decays smoothly with cosine decay.

Why warmup?

- Early Transformer training can be unstable.
- Warmup prevents large updates before activations and gradients are well-scaled.

Why cosine?

- It gradually lowers the learning rate without abrupt drops.
- It is a common stable default for pretraining/fine-tuning.

### 16.5 Gradient Accumulation

The Colab T4 config uses:

```yaml
batch_size: 10
grad_accum_steps: 6
max_steps: 1000
warmup_steps: 100
```

This means effective batch size is approximately:

```text
10 x 6 = 60 examples per optimizer step
```

Why?

- A full 110M/124M model may not fit a large batch on one GPU.
- Gradient accumulation simulates a larger batch using smaller microbatches.
- This setting uses more of a single T4's memory than the earlier 8-by-8 setup while keeping the effective batch size nearly the same.

### 16.6 Mixed Precision

The code supports:

- `fp32`
- `fp16`
- `bf16`

The strict config uses `fp16`.

Why mixed precision?

- Saves GPU memory.
- Speeds up matrix operations on supported GPUs.
- Enables larger models or larger batches.

For FP16, gradient scaling is used to reduce underflow risk.

### 16.7 Gradient Clipping

The code clips gradient norm:

```yaml
grad_clip: 1.0
```

Why?

- It prevents exploding gradients.
- It is especially useful during early training and warm-start fine-tuning.

## 17. Checkpointing and Logging

Checkpoints are saved with:

- Model state dict
- Optimizer state dict
- Step number
- GradScaler state
- Metadata

Metadata includes:

- Seed
- Source tokenizer path
- Target tokenizer path
- Total parameter count
- Trainable parameter count

Metrics are logged in two formats:

- JSONL for flexible programmatic reading
- CSV for easy spreadsheet inspection

Logged fields:

- time
- stage
- split
- step
- loss
- BLEU
- chrF
- learning rate
- batch size

## 18. Evaluation

The evaluation path computes:

- Cross-entropy loss
- BLEU
- chrF++

### 18.1 BLEU

BLEU measures n-gram overlap between generated translations and reference translations.

Strength:

- Common MT metric.
- Easy to compare across systems.

Weakness:

- Can be harsh for valid paraphrases.
- Less sensitive to morphology than character-level metrics.

### 18.2 chrF++

chrF++ is a character n-gram F-score with word-order support.

Why useful here?

- Marathi is morphologically rich.
- Character-level similarity can capture partial correctness in inflected words.
- It can be more informative than BLEU for Indic language translation.

Interview answer:

> I report both BLEU and chrF++. BLEU gives standard MT comparability, while chrF++ is helpful for morphologically rich languages where small suffix changes matter.

### 18.3 Decoding

The evaluation uses greedy decoding.

Greedy decoding:

- Picks the highest-probability token at each step.
- Is simple and deterministic.
- Is fast for evaluation.

Limitation:

- Beam search may produce better translations.
- Sampling is not appropriate for deterministic translation evaluation.

## 19. Plotting

The plot stage reads metric rows and writes:

- `loss.png`
- `bleu_100.png`
- `chrfpp_100.png`

Plots help answer:

- Is training loss decreasing?
- Is validation loss improving?
- Is the model overfitting?
- Are BLEU and chrF++ improving over time?

## 20. Parameter Count Rationale

The tests assert exact parameter counts for the strict model sizes.

### 20.1 Encoder

Config:

```yaml
vocab_size: 45000
d_model: 768
n_layers: 12
n_heads: 12
n_kv_heads: 4
ffn_dim: 3072
```

Expected parameters:

```text
110,076,672
```

Why this reaches about 110M:

- Embedding table is large: `45000 x 768`.
- 12 Transformer blocks dominate the remaining parameters.
- GQA reduces K/V projection parameters compared to full MHA.
- Tied embeddings avoid a separate output projection matrix.

### 20.2 Decoder

Config:

```yaml
vocab_size: 50257
d_model: 768
n_layers: 12
n_heads: 12
n_kv_heads: 4
ffn_dim: 3584
```

Expected parameters:

```text
123,551,232
```

Why the decoder FFN is 3584:

- The decoder vocabulary is large.
- GQA reduces attention parameters.
- The assignment asks for approximately 124M parameters.
- Increasing the FFN dimension from 3072 to 3584 brings the decoder closer to the target while preserving a standard Transformer shape.

Interview answer:

> I controlled parameter count mainly through vocabulary size, layer count, hidden size, number of heads, KV heads, and FFN dimension. The tests verify the final encoder and decoder counts exactly.

## 21. Why These Design Choices?

### 21.1 Why Not RNN/LSTM?

Transformers are better suited here because:

- They parallelize across tokens during training.
- Attention directly models long-range dependencies.
- Modern MT systems are Transformer-based.
- The assignment specifically asks for modern Transformer features.

### 21.2 Why Encoder-Decoder Instead of Decoder-Only Translation?

A decoder-only model could concatenate source and target text, but encoder-decoder is cleaner for translation:

- Encoder specializes in source understanding.
- Decoder specializes in target generation.
- Cross-attention gives explicit source conditioning.
- It is the classical and still strong architecture for MT.

### 21.3 Why Pretrain Separately?

Separate pretraining helps because:

- Encoder learns Hindi context before translation.
- Decoder learns Marathi fluency before translation.
- Translation fine-tuning then focuses on alignment between source and target.

### 21.4 Why No NSP?

The encoder uses MLM only.

Reason:

- NSP is not necessary for this translation pipeline.
- The downstream task needs token/sentence representation quality more than binary sentence-pair classification.
- MLM is the main useful BERT-style objective for source encoding.

### 21.5 Why Weight Tying?

The encoder and decoder tie embedding weights with the LM head.

Benefits:

- Reduces parameter count.
- Common in language models.
- Encourages input and output token representations to share structure.

## 22. What Happens During Each Stage?

### 22.1 Stage `mlm`

Input:

- Hindi monolingual data

Output:

- Hindi encoder checkpoint
- MLM training/validation loss logs

Purpose:

- Teach encoder bidirectional Hindi representations.

### 22.2 Stage `clm`

Input:

- Marathi monolingual data

Output:

- Marathi decoder checkpoint
- CLM training/validation loss logs

Purpose:

- Teach decoder fluent Marathi generation.

### 22.3 Stage `mt`

Input:

- Parallel Hindi-Marathi data
- Optional encoder checkpoint
- Optional decoder checkpoint

Output:

- MT checkpoint
- Train/valid loss
- BLEU
- chrF++

Purpose:

- Learn translation alignment and generation conditioned on source text.

### 22.4 Stage `eval`

Input:

- Test split
- MT checkpoint path in config

Output:

- Final BLEU and chrF++

### 22.5 Stage `plot`

Input:

- Metric JSONL files

Output:

- Loss, BLEU, and chrF++ plots

## 23. Code Flow for Translation

This is the most important flow to explain in an interview.

### 23.1 Training-Time Flow

1. `ParallelDataset` returns one Hindi sequence and one Marathi sequence.
2. `make_mt_batch` pads both sides.
3. Target sequence is shifted:
   - decoder input gets all tokens except the last
   - labels get all tokens except the first
4. `Seq2SeqModel.forward` runs the encoder on Hindi.
5. Encoder output is passed into the MT decoder.
6. The decoder uses:
   - causal self-attention over Marathi prefix
   - cross-attention over Hindi encoder output
7. Final logits predict next Marathi tokens.
8. Cross-entropy trains against labels.

### 23.2 Inference-Time Flow

1. Encode the full Hindi source once.
2. Start Marathi output with BOS.
3. Run decoder to predict next token.
4. Append the predicted token.
5. Repeat until EOS or max length.

The current code uses greedy decoding.

## 24. Important Implementation Details

### 24.1 Padding Mask

Padding mask shape becomes:

```text
batch x 1 x 1 x sequence_length
```

This broadcasts across heads and query positions.

### 24.2 Causal Mask

Causal mask shape becomes:

```text
1 x 1 x target_length x target_length
```

It blocks future target tokens.

### 24.3 Combined Decoder Mask

The decoder combines:

- causal mask
- target padding mask

This prevents both future-token leakage and padding attention.

### 24.4 `-100` Labels

PyTorch cross-entropy ignores label `-100`.

Used for:

- Unmasked MLM positions
- Padding tokens
- Last CLM position where no next token exists

## 25. Tests

The test file covers three important risks.

### 25.1 Parameter Count Test

It builds the strict encoder and decoder on the meta device and checks exact parameter counts.

Why meta device?

- It avoids allocating full model memory.
- It allows fast parameter counting.

### 25.2 Shape Test

It checks:

- Encoder MLM output shape
- Decoder CLM output shape
- Seq2Seq MT output shape

This catches vocabulary mismatch and batch/sequence errors.

### 25.3 Plot Test

It creates fake metrics and checks that plot files are generated.

This verifies reporting artifacts required by the assignment.

## 26. Current Limitations and Honest Discussion

This is useful in interviews because strong candidates can explain trade-offs honestly.

### 26.1 Greedy Decoding Only

The code uses greedy decoding, which is simple but not always best.

Possible improvement:

- Add beam search with length penalty.

### 26.2 No KV Cache

The decoder recomputes previous tokens during generation.

Possible improvement:

- Implement KV caching for faster autoregressive decoding.

### 26.3 No Label Smoothing

The loss uses standard cross-entropy.

Possible improvement:

- Add label smoothing for MT to reduce overconfidence.

### 26.4 No DDP/FSDP

The code is single-process training.

Possible improvement:

- Add DistributedDataParallel for multi-GPU training.

### 26.5 No Advanced Data Filtering

The dataset reader assumes parallel lines are aligned.

Possible improvement:

- Add length-ratio filtering, duplicate removal, Unicode normalization, and language ID filtering.

### 26.6 Max Sequence Length is 128 in Strict Config

The strict config uses `max_seq_len: 128`.

Reason:

- Practical for T4-style memory constraints.
- Keeps training feasible.

Possible improvement:

- Increase max length if more GPU memory is available.

### 26.7 Cross-Attention Starts Randomly

Warm-started decoder self-attention and FFN weights are useful, but cross-attention is new.

Possible improvement:

- Use staged fine-tuning:
  - first freeze pretrained weights and train cross-attention
  - then unfreeze everything

## 27. Development Disclosure

### 27.1 GPU Hardware Used

Training and debugging support both Google Colab single-T4 execution and Kaggle-style multi-GPU execution. Colab uses `configs/colab_t4.yaml` with single-process CUDA training and model-only checkpoints. Kaggle multi-GPU remains available by explicitly launching with `GPUS=2`.

### 27.2 LLM Assistance Used

GitHub Copilot Chat was used during development for code assistance, report editing, and debugging support. All final code and report content were reviewed and adapted for the submission.

## 28. Interview Q&A

### Q1. What exactly did you build?

I built a from-scratch Hindi-to-Marathi Transformer pipeline. It has a BERT-like Hindi encoder pretrained with MLM, a GPT-style Marathi decoder pretrained with CLM, and a warm-started encoder-decoder translation model with cross-attention.

### Q2. Why did you use an encoder-decoder model?

Translation naturally has a source sequence and a target sequence. The encoder reads the full Hindi sentence bidirectionally, and the decoder generates Marathi autoregressively while attending to the Hindi encoder states.

### Q3. Why MLM for Hindi?

MLM trains bidirectional understanding. Since the source sentence is fully available during translation, the encoder should use both left and right context.

### Q4. Why CLM for Marathi?

The target side is generated left to right. CLM matches inference, where the decoder predicts the next Marathi token from previously generated tokens.

### Q5. What is cross-attention?

Cross-attention lets decoder states attend to encoder outputs. Queries come from the decoder, while keys and values come from the encoder.

### Q6. Why is decoder checkpoint loading non-strict?

The pretrained CLM decoder has self-attention and FFN layers but no cross-attention. The MT decoder adds cross-attention layers, so some new weights are missing from the CLM checkpoint and must be randomly initialized.

### Q7. What is GQA?

Grouped Query Attention uses more query heads than key/value heads. In this project, 12 query heads share 4 key/value heads. This reduces KV projection size while preserving multiple query heads.

### Q8. Why not use full MHA?

Full MHA is valid but more expensive. GQA is a modern compromise that improves memory efficiency, especially useful for autoregressive decoding.

### Q9. What is RoPE?

RoPE rotates query and key vectors based on position. This injects positional information directly into attention scores and gives relative-position awareness without learned positional embeddings.

### Q10. Why not learned positional embeddings?

Learned positional embeddings add parameters and are tied to a fixed maximum position. RoPE is parameter-free and widely used in modern Transformer architectures.

### Q11. Why RMSNorm instead of LayerNorm?

RMSNorm normalizes by root mean square without subtracting the mean. It is simpler, efficient, and common in modern LLM-style Transformers.

### Q12. Why use pre-norm blocks?

Pre-norm improves training stability because the residual path carries gradients more directly through the network.

### Q13. Why SentencePiece?

SentencePiece works directly with raw Unicode and learns subwords. That is useful for Hindi and Marathi because they are morphologically rich and use complex scripts.

### Q14. Why separate tokenizers?

Separate tokenizers let the Hindi encoder and Marathi decoder specialize and help hit exact parameter targets. A shared tokenizer is also reasonable for related languages, but the final config prioritizes parameter control.

### Q15. How do you prevent the decoder from seeing future tokens?

The decoder creates an upper-triangular causal mask. Future positions get a large negative attention score before softmax, so their probability becomes near zero.

### Q16. How do you handle padding?

The code creates attention masks from non-padding tokens. Padding positions are converted to large negative additive masks. Loss also ignores padding labels using `-100`.

### Q17. What does `ignore_index=-100` do?

It tells PyTorch cross-entropy not to compute loss for those positions. This is used for padding, unmasked MLM tokens, and invalid next-token positions.

### Q18. What is teacher forcing?

During MT training, the decoder receives the gold target prefix and predicts the next target token. This trains all positions in parallel and stabilizes learning.

### Q19. Why tie embeddings?

Tying embeddings shares the input embedding matrix with the output LM head. This reduces parameters and often improves language modeling generalization.

### Q20. Why use AdamW?

AdamW is standard for Transformer training because it handles adaptive updates well and decouples weight decay from gradient updates.

### Q21. Why use warmup?

Warmup avoids unstable large updates at the beginning of training, when weights and activations are not calibrated.

### Q22. Why cosine decay?

Cosine decay smoothly reduces the learning rate over training and is a strong default for pretraining and fine-tuning.

### Q23. What are BLEU and chrF++?

BLEU measures word/subword n-gram overlap with references. chrF++ measures character-level overlap with word-order support, which is useful for morphologically rich languages.

### Q24. Why is chrF++ important for Marathi?

Marathi has rich morphology. A generated word may be close at the character or suffix level even if exact word n-gram matching is imperfect, so chrF++ can show partial correctness better than BLEU.

### Q25. What is the biggest weakness of the current solution?

The main practical limitations are greedy decoding, no KV cache, and limited data preprocessing. These are engineering improvements that can be added without changing the core architecture.

### Q26. How would you improve translation quality next?

I would add stronger data filtering, train longer, add beam search, tune tokenizer vocabulary, add label smoothing, and possibly use staged fine-tuning where cross-attention is trained first before unfreezing the entire model.

### Q27. How do you know parameter targets are met?

`tests/test_part2.py` creates the strict encoder and decoder on the meta device and asserts exact counts: 110,076,672 for the encoder and 123,551,232 for the decoder.

### Q28. Why does the MT model have more parameters than the standalone decoder?

The MT decoder includes cross-attention layers in addition to the pretrained decoder layers. Those cross-attention weights are necessary for source conditioning.

### Q29. Why is RoPE disabled in cross-attention?

Source and target positions are not the same timeline. RoPE is cleanest for self-attention inside one sequence. Cross-attention already receives source order through encoder representations.

### Q30. What does auto batch-size tuning do?

On CUDA, it tries larger microbatch sizes until an out-of-memory error occurs, then keeps the largest working size. This helps use GPU memory efficiently.

## 29. Suggested Interview Walkthrough

A strong explanation order:

1. Start with the objective: Hindi-to-Marathi translation.
2. Explain the three-stage training design: MLM, CLM, MT.
3. Explain the architecture: encoder, decoder, cross-attention.
4. Explain modern block choices: RoPE, GQA, RMSNorm.
5. Explain training mechanics: masks, losses, teacher forcing, warm-starting.
6. Explain evaluation: BLEU, chrF++, loss plots.
7. End with limitations and future improvements.

Short version:

> The system first learns Hindi understanding through MLM and Marathi fluency through CLM. Then I connect them with cross-attention and fine-tune on parallel Hindi-Marathi pairs. The Transformer blocks use RoPE for positional encoding, GQA for efficient attention, and RMSNorm for stable modern pre-norm training.

## 30. Files to Mention During Interview

If asked where something is implemented:

| Topic | File |
|---|---|
| Model architecture | `src/models/transformer.py` |
| GQA | `src/models/attention.py` |
| RoPE | `src/models/rope.py` |
| RMSNorm | `src/models/norms.py` |
| Tokenizer training | `src/data/tokenizer.py` |
| MLM/CLM/MT batches | `src/data/datasets.py` |
| Training loops | `scripts/train.py` |
| BLEU/chrF++ | `src/utils/metrics.py` |
| Greedy decoding | `src/utils/decoding.py` |
| LR schedule | `src/utils/schedule.py` |
| Parameter count tests | `tests/test_part2.py` |

## 31. Final Defense Statement

This solution is defensible because it implements the required architecture directly rather than wrapping a pretrained library model. It shows understanding of both language-model pretraining and sequence-to-sequence translation. The encoder is trained for source understanding, the decoder is trained for target fluency, and the MT model combines both through cross-attention.

The key rationale is:

- MLM is appropriate for bidirectional Hindi source encoding.
- CLM is appropriate for autoregressive Marathi generation.
- Cross-attention is the correct mechanism to condition generation on the source sentence.
- RoPE, GQA, and RMSNorm are modern Transformer upgrades that improve positional modeling, efficiency, and stability.
- SentencePiece is appropriate for Indic subword tokenization.
- BLEU and chrF++ together provide a better evaluation picture than either metric alone.

In one sentence:

> I built a modern from-scratch Transformer translation pipeline where Hindi understanding, Marathi generation, and cross-lingual alignment are learned in separate but connected stages.
