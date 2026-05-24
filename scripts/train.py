import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, DistributedSampler

from src.data.datasets import (
    MonolingualDataset,
    ParallelDataset,
    make_clm_batch,
    make_mlm_batch,
    make_mt_batch,
)
from src.data.tokenizer import (
    SentencePieceTokenizer,
    train_sentencepiece,
    train_sentencepiece_from_files,
)
from src.models.transformer import DecoderModel, EncoderModel, Seq2SeqModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.decoding import greedy_decode
from src.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_local_rank,
    synchronize,
    reduce_loss,
    get_device as get_distributed_device,
)
from src.utils.metrics import compute_bleu_chrf
from src.utils.params import count_parameters, count_trainable_parameters
from src.utils.schedule import get_lr


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    # Use the distributed device getter if available
    return get_distributed_device()


def get_amp_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def make_scaler(device, amp_dtype):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def move_to_device(items, device):
    return [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in items]


def model_cfg(cfg, role):
    if "models" in cfg:
        out = dict(cfg["models"][role])
    else:
        out = dict(cfg["model"])
    if "ffn_dim" not in out:
        out["ffn_dim"] = out["d_model"] * out.get("ffn_mult", 4)
    out.pop("ffn_mult", None)
    return out


def output_dir(cfg, stage):
    root = cfg["training"].get("output_dir", "checkpoints")
    path = os.path.join(root, stage)
    os.makedirs(path, exist_ok=True)
    return path


def save_run_config(cfg, out_dir):
    path = os.path.join(out_dir, "resolved_config.yaml")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def data_path(cfg, split, lang):
    return os.path.join(cfg["data"]["data_dir"], f"{split}.{lang}")


def tokenizer_section(cfg, role):
    if "tokenizers" in cfg:
        return cfg["tokenizers"][role]
    return cfg["tokenizer"]


def prepare_tokenizers(cfg):
    data_dir = cfg["data"]["data_dir"]
    src_lang = cfg["data"]["src_lang"]
    tgt_lang = cfg["data"]["tgt_lang"]
    world_size = get_world_size()
    main_process = is_main_process()

    # In distributed jobs, rank 0 prepares the tokenizer files once and the
    # other ranks wait until the files are ready.
    if world_size > 1 and not main_process:
        synchronize()

    if "tokenizers" not in cfg or cfg.get("tokenizers", {}).get("shared", False):
        tok_cfg = cfg.get("tokenizer", cfg.get("tokenizers", {}).get("shared_config", {}))
        model_path = tok_cfg["model_prefix"] + ".model"
        if main_process:
            model_path = train_sentencepiece(
                data_dir,
                tok_cfg["model_prefix"],
                tok_cfg["vocab_size"],
                tok_cfg["character_coverage"],
            )
        if world_size > 1:
            synchronize()
        tokenizer = SentencePieceTokenizer(model_path)
        return tokenizer, tokenizer

    src_cfg = tokenizer_section(cfg, "src")
    tgt_cfg = tokenizer_section(cfg, "tgt")
    src_model = src_cfg["model_prefix"] + ".model"
    tgt_model = tgt_cfg["model_prefix"] + ".model"
    if main_process:
        src_model = train_sentencepiece_from_files(
            [data_path(cfg, "train", src_lang)],
            src_cfg["model_prefix"],
            src_cfg["vocab_size"],
            src_cfg["character_coverage"],
            src_cfg.get("model_type", "unigram"),
            src_cfg.get("input_sentence_size", 0),
            src_cfg.get("max_sentence_length", 4192),
        )
        tgt_model = train_sentencepiece_from_files(
            [data_path(cfg, "train", tgt_lang)],
            tgt_cfg["model_prefix"],
            tgt_cfg["vocab_size"],
            tgt_cfg["character_coverage"],
            tgt_cfg.get("model_type", "unigram"),
            tgt_cfg.get("input_sentence_size", 0),
            tgt_cfg.get("max_sentence_length", 4192),
        )
    if world_size > 1:
        synchronize()
    return SentencePieceTokenizer(src_model), SentencePieceTokenizer(tgt_model)


def make_loader(dataset, cfg, batch_size, shuffle):
    num_workers = cfg["training"].get("num_workers", 0)
    world_size = get_world_size()
    rank = get_rank()
    
    # Use DistributedSampler if in distributed mode
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=cfg["training"].get("seed", 42),
        )
        # When using sampler, shuffle must be False in DataLoader
        kwargs = {
            "batch_size": batch_size,
            "sampler": sampler,
            "num_workers": num_workers,
            "collate_fn": lambda x: x,
            "pin_memory": torch.cuda.is_available(),
        }
    else:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": lambda x: x,
            "pin_memory": torch.cuda.is_available(),
        }
    
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg["training"].get("prefetch_factor", 2)
    
    return DataLoader(dataset, **kwargs)


def make_optimizer(model, cfg, device):
    kwargs = {
        "lr": cfg["training"]["lr"],
        "weight_decay": cfg["training"]["weight_decay"],
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def maybe_compile(model, cfg, device):
    if device.type == "cuda" and cfg["training"].get("compile", False):
        try:
            return torch.compile(model)
        except Exception as exc:
            print(f"[compile] disabled: {exc}")
    return model


def unwrap_model(model):
    """Unwrap model from DDP or torch.compile wrappers."""
    # Handle DDP wrapper
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # Handle torch.compile wrapper
    return getattr(model, "_orig_mod", model)


def wrap_parallel_model(model, cfg, device):
    if get_world_size() > 1:
        local_rank = get_local_rank()
        if device.type == "cuda":
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        return torch.nn.parallel.DistributedDataParallel(model)

    if (
        cfg.get("distributed", {}).get("enabled", False)
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        device_ids = list(range(torch.cuda.device_count()))
        print(f"[parallel] using DataParallel across GPUs {device_ids}")
        return torch.nn.DataParallel(model, device_ids=device_ids)

    return model


def fixed_subset(dataset, sample_size, seed):
    if sample_size is None or sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), sample_size)
    return Subset(dataset, indices)


def auto_tune_batch_size(make_test_loader, test_step, cfg, device):
    start_bs = cfg["training"]["batch_size"]
    if device.type != "cuda" or not cfg["training"].get("auto_batch_size", False):
        return start_bs

    max_bs = cfg["training"].get("max_auto_batch_size", start_bs)
    best = None
    bs = start_bs
    while bs <= max_bs:
        try:
            loader = make_test_loader(bs)
            batch = next(iter(loader))
            test_step(batch)
            best = bs
            bs *= 2
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            break

    if best is not None:
        print(f"[batch] auto-tuned microbatch={best}")
        return best

    bs = max(1, start_bs // 2)
    while bs >= 1:
        try:
            loader = make_test_loader(bs)
            batch = next(iter(loader))
            test_step(batch)
            print(f"[batch] auto-tuned microbatch={bs}")
            return bs
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            bs //= 2
    raise RuntimeError("Unable to find a working batch size")


class MetricLogger:
    def __init__(self, out_dir, stage):
        self.jsonl_path = os.path.join(out_dir, f"{stage}_metrics.jsonl")
        self.csv_path = os.path.join(out_dir, f"{stage}_metrics.csv")
        self.fields = ["time", "stage", "split", "step", "loss", "bleu", "chrf", "lr", "batch_size"]
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def log(self, **row):
        record = {field: row.get(field) for field in self.fields}
        record["time"] = time.time()
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(record)


def train_step_end(model, optimizer, scaler, cfg):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def set_lr(optimizer, step, cfg):
    lr = get_lr(step, cfg["training"]["lr"], cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_encoder(cfg, tokenizer):
    arch = model_cfg(cfg, "encoder")
    arch["vocab_size"] = tokenizer.sp.get_piece_size()
    return EncoderModel(**arch)


def build_decoder(cfg, tokenizer):
    arch = model_cfg(cfg, "decoder")
    arch["vocab_size"] = tokenizer.sp.get_piece_size()
    return DecoderModel(**arch)


def build_mt_model(cfg, src_tokenizer, tgt_tokenizer):
    enc_cfg = model_cfg(cfg, "encoder")
    dec_cfg = model_cfg(cfg, "decoder")
    enc_cfg["vocab_size"] = src_tokenizer.sp.get_piece_size()
    dec_cfg["vocab_size"] = tgt_tokenizer.sp.get_piece_size()
    return Seq2SeqModel(encoder_config=enc_cfg, decoder_config=dec_cfg)


def load_warm_start(cfg, model, device):
    mt_cfg = cfg.get("mt", {})
    enc_ckpt = mt_cfg.get("encoder_checkpoint")
    dec_ckpt = mt_cfg.get("decoder_checkpoint")
    
    # Convert relative paths to absolute paths (relative to repo_root for consistency)
    if enc_ckpt and not os.path.isabs(enc_ckpt):
        enc_ckpt = os.path.join(repo_root, enc_ckpt)
    if dec_ckpt and not os.path.isabs(dec_ckpt):
        dec_ckpt = os.path.join(repo_root, dec_ckpt)
    
    if enc_ckpt:
        load_checkpoint(enc_ckpt, model.encoder, map_location=device)
        print(f"[warm-start] loaded encoder: {enc_ckpt}")
    if dec_ckpt:
        ckpt = torch.load(dec_ckpt, map_location=device)
        missing, unexpected = model.decoder.load_state_dict(ckpt["model"], strict=False)
        cross_missing = [k for k in missing if "cross_" in k]
        other_missing = [k for k in missing if "cross_" not in k]
        if other_missing or unexpected:
            print(f"[warm-start] decoder non-cross missing={other_missing} unexpected={unexpected}")
        print(f"[warm-start] loaded decoder: {dec_ckpt}; initialized {len(cross_missing)} cross-attention tensors")

    if mt_cfg.get("freeze_pretrained", False):
        for name, param in model.named_parameters():
            if "cross_" not in name:
                param.requires_grad = False


def compute_mlm_loss(model, batch, tokenizer, device, amp_dtype):
    input_ids, attn_mask, labels = make_mlm_batch(batch, tokenizer)
    input_ids, attn_mask, labels = move_to_device([input_ids, attn_mask, labels], device)
    with autocast_context(device, amp_dtype):
        logits = model.forward_mlm(input_ids, attn_mask=attn_mask)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def compute_clm_loss(model, batch, tokenizer, device, amp_dtype):
    input_ids, attn_mask, labels = make_clm_batch(batch, tokenizer)
    input_ids, attn_mask, labels = move_to_device([input_ids, attn_mask, labels], device)
    with autocast_context(device, amp_dtype):
        logits = model(input_ids, tgt_mask=attn_mask)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def compute_mt_loss(model, batch, src_tokenizer, tgt_tokenizer, device, amp_dtype):
    src_ids, src_mask, tgt_in, tgt_mask, labels = make_mt_batch(batch, src_tokenizer, tgt_tokenizer)
    src_ids, src_mask, tgt_in, tgt_mask, labels = move_to_device([src_ids, src_mask, tgt_in, tgt_mask, labels], device)
    with autocast_context(device, amp_dtype):
        logits = model(src_ids, src_mask, tgt_in, tgt_mask=tgt_mask)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def evaluate_loss(model, dataset, batch_size, cfg, loss_fn, max_batches=None):
    device = next(model.parameters()).device
    amp_dtype = get_amp_dtype(cfg["training"].get("amp_dtype", "fp16"))
    loader = make_loader(dataset, cfg, batch_size, shuffle=False)
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            losses.append(float(loss_fn(model, batch, device, amp_dtype).item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def evaluate_mt_metrics(model, dataset, src_tokenizer, tgt_tokenizer, cfg, batch_size, max_samples):
    device = next(model.parameters()).device
    max_len = model_cfg(cfg, "decoder")["max_seq_len"]
    subset = fixed_subset(dataset, max_samples, cfg["training"]["seed"])
    loader = make_loader(subset, cfg, batch_size, shuffle=False)
    preds = []
    refs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            src_ids, src_mask, _, _, labels = make_mt_batch(batch, src_tokenizer, tgt_tokenizer)
            src_ids, src_mask = move_to_device([src_ids, src_mask], device)
            out_ids = greedy_decode(
                model,
                src_ids,
                src_mask,
                tgt_tokenizer.bos_id,
                tgt_tokenizer.eos_id,
                max_len,
                device,
            )
            for ids in out_ids[:, 1:].tolist():
                if tgt_tokenizer.eos_id in ids:
                    ids = ids[: ids.index(tgt_tokenizer.eos_id)]
                preds.append(tgt_tokenizer.decode(ids))
            for lbl in labels.tolist():
                lbl = [t for t in lbl if t != -100 and t != tgt_tokenizer.pad_id]
                refs.append(tgt_tokenizer.decode(lbl))
    model.train()
    return compute_bleu_chrf(preds, refs)


def train_mlm(cfg):
    # Setup distributed training
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = get_device()
    
    if is_main_process():
        print(f"[mlm] rank={rank} world_size={world_size} local_rank={local_rank}")
    
    amp_dtype = get_amp_dtype(cfg["training"].get("amp_dtype", "fp16"))
    out_dir = output_dir(cfg, "mlm")
    
    # Only save config from rank 0
    if is_main_process():
        save_run_config(cfg, out_dir)
    
    logger = MetricLogger(out_dir, "mlm") if is_main_process() else None

    src_tokenizer, _ = prepare_tokenizers(cfg)
    train_dataset = MonolingualDataset(data_path(cfg, "train", cfg["data"]["src_lang"]), src_tokenizer, model_cfg(cfg, "encoder")["max_seq_len"])
    valid_dataset = MonolingualDataset(data_path(cfg, "test", cfg["data"]["src_lang"]), src_tokenizer, model_cfg(cfg, "encoder")["max_seq_len"])

    model = build_encoder(cfg, src_tokenizer).to(device)
    model = maybe_compile(model, cfg, device)
    model = wrap_parallel_model(model, cfg, device)
    
    if is_main_process():
        print(f"[mlm] params={count_parameters(unwrap_model(model))/1e6:.2f}M trainable={count_trainable_parameters(unwrap_model(model))/1e6:.2f}M")

    optimizer = make_optimizer(model, cfg, device)
    scaler = make_scaler(device, amp_dtype)
    optimizer.zero_grad(set_to_none=True)

    def tune_step(batch):
        loss = compute_mlm_loss(model, batch, src_tokenizer, device, amp_dtype)
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)

    # Auto-tune only on rank 0
    if is_main_process():
        batch_size = auto_tune_batch_size(lambda bs: make_loader(train_dataset, cfg, bs, True), tune_step, cfg, device)
    else:
        batch_size = cfg["training"]["batch_size"]
    
    # Broadcast batch size to all ranks
    if world_size > 1:
        batch_size_tensor = torch.tensor(batch_size, device=device)
        torch.distributed.broadcast(batch_size_tensor, src=0)
        batch_size = int(batch_size_tensor.item())
    
    loader = make_loader(train_dataset, cfg, batch_size, True)
    grad_accum = cfg["training"]["grad_accum_steps"]
    opt_step = 0
    micro_step = 0
    last_loss = 0.0
    model.train()

    while opt_step < cfg["training"]["max_steps"]:
        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(opt_step // cfg["training"]["max_steps"])
        
        for batch in loader:
            if opt_step >= cfg["training"]["max_steps"]:
                break
            lr = set_lr(optimizer, opt_step + 1, cfg)
            loss = compute_mlm_loss(model, batch, src_tokenizer, device, amp_dtype)
            last_loss = float(loss.item())
            scaler.scale(loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue
            opt_step += 1
            train_step_end(model, optimizer, scaler, cfg)
            
            # Reduce loss across ranks
            if world_size > 1:
                loss_dict = reduce_loss({"loss": last_loss})
                last_loss = loss_dict["loss"]
            
            if opt_step % cfg["training"]["log_every"] == 0 and is_main_process():
                logger.log(stage="mlm", split="train", step=opt_step, loss=last_loss, lr=lr, batch_size=batch_size)
                print(f"[mlm] step={opt_step} loss={last_loss:.4f} lr={lr:.2e} bs={batch_size}")
            
            if opt_step % cfg["training"]["eval_every"] == 0:
                max_batches = cfg["evaluation"].get("loss_batches", 20)
                val_loss = evaluate_loss(
                    model,
                    valid_dataset,
                    cfg["evaluation"].get("loss_batch_size", batch_size),
                    cfg,
                    lambda m, b, d, a: compute_mlm_loss(m, b, src_tokenizer, d, a),
                    max_batches=max_batches,
                )
                if is_main_process():
                    logger.log(stage="mlm", split="valid", step=opt_step, loss=val_loss, lr=lr, batch_size=batch_size)
                    print(f"[mlm:valid] step={opt_step} loss={val_loss:.4f}")
            
            if opt_step % cfg["training"]["save_every"] == 0 and is_main_process():
                ckpt = os.path.join(out_dir, f"encoder_mlm_step{opt_step}.pt")
                raw_model = unwrap_model(model)
                save_checkpoint(ckpt, raw_model, optimizer, opt_step, scaler, metadata=checkpoint_metadata(cfg, src_tokenizer, None, raw_model))
    
    if is_main_process():
        raw_model = unwrap_model(model)
        ckpt = os.path.join(out_dir, f"encoder_mlm_final_step{opt_step}.pt")
        save_checkpoint(ckpt, raw_model, optimizer, opt_step, scaler, metadata=checkpoint_metadata(cfg, src_tokenizer, None, raw_model))


def train_clm(cfg):
    # Setup distributed training
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = get_device()
    
    if is_main_process():
        print(f"[clm] rank={rank} world_size={world_size} local_rank={local_rank}")
    
    amp_dtype = get_amp_dtype(cfg["training"].get("amp_dtype", "fp16"))
    out_dir = output_dir(cfg, "clm")
    
    # Only save config from rank 0
    if is_main_process():
        save_run_config(cfg, out_dir)
    
    logger = MetricLogger(out_dir, "clm") if is_main_process() else None

    _, tgt_tokenizer = prepare_tokenizers(cfg)
    train_dataset = MonolingualDataset(data_path(cfg, "train", cfg["data"]["tgt_lang"]), tgt_tokenizer, model_cfg(cfg, "decoder")["max_seq_len"])
    valid_dataset = MonolingualDataset(data_path(cfg, "test", cfg["data"]["tgt_lang"]), tgt_tokenizer, model_cfg(cfg, "decoder")["max_seq_len"])

    model = build_decoder(cfg, tgt_tokenizer).to(device)
    model = maybe_compile(model, cfg, device)
    model = wrap_parallel_model(model, cfg, device)
    
    if is_main_process():
        print(f"[clm] params={count_parameters(unwrap_model(model))/1e6:.2f}M trainable={count_trainable_parameters(unwrap_model(model))/1e6:.2f}M")

    optimizer = make_optimizer(model, cfg, device)
    scaler = make_scaler(device, amp_dtype)
    optimizer.zero_grad(set_to_none=True)

    def tune_step(batch):
        loss = compute_clm_loss(model, batch, tgt_tokenizer, device, amp_dtype)
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)

    # Auto-tune only on rank 0
    if is_main_process():
        batch_size = auto_tune_batch_size(lambda bs: make_loader(train_dataset, cfg, bs, True), tune_step, cfg, device)
    else:
        batch_size = cfg["training"]["batch_size"]
    
    # Broadcast batch size to all ranks
    if world_size > 1:
        batch_size_tensor = torch.tensor(batch_size, device=device)
        torch.distributed.broadcast(batch_size_tensor, src=0)
        batch_size = int(batch_size_tensor.item())
    
    loader = make_loader(train_dataset, cfg, batch_size, True)
    grad_accum = cfg["training"]["grad_accum_steps"]
    opt_step = 0
    micro_step = 0
    last_loss = 0.0
    model.train()

    while opt_step < cfg["training"]["max_steps"]:
        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(opt_step // cfg["training"]["max_steps"])
        
        for batch in loader:
            if opt_step >= cfg["training"]["max_steps"]:
                break
            lr = set_lr(optimizer, opt_step + 1, cfg)
            loss = compute_clm_loss(model, batch, tgt_tokenizer, device, amp_dtype)
            last_loss = float(loss.item())
            scaler.scale(loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue
            opt_step += 1
            train_step_end(model, optimizer, scaler, cfg)
            
            # Reduce loss across ranks
            if world_size > 1:
                loss_dict = reduce_loss({"loss": last_loss})
                last_loss = loss_dict["loss"]
            
            if opt_step % cfg["training"]["log_every"] == 0 and is_main_process():
                logger.log(stage="clm", split="train", step=opt_step, loss=last_loss, lr=lr, batch_size=batch_size)
                print(f"[clm] step={opt_step} loss={last_loss:.4f} lr={lr:.2e} bs={batch_size}")
            
            if opt_step % cfg["training"]["eval_every"] == 0:
                val_loss = evaluate_loss(
                    model,
                    valid_dataset,
                    cfg["evaluation"].get("loss_batch_size", batch_size),
                    cfg,
                    lambda m, b, d, a: compute_clm_loss(m, b, tgt_tokenizer, d, a),
                    max_batches=cfg["evaluation"].get("loss_batches", 20),
                )
                if is_main_process():
                    logger.log(stage="clm", split="valid", step=opt_step, loss=val_loss, lr=lr, batch_size=batch_size)
                    print(f"[clm:valid] step={opt_step} loss={val_loss:.4f}")
            
            if opt_step % cfg["training"]["save_every"] == 0 and is_main_process():
                ckpt = os.path.join(out_dir, f"decoder_clm_step{opt_step}.pt")
                raw_model = unwrap_model(model)
                save_checkpoint(ckpt, raw_model, optimizer, opt_step, scaler, metadata=checkpoint_metadata(cfg, None, tgt_tokenizer, raw_model))
    
    if is_main_process():
        raw_model = unwrap_model(model)
        ckpt = os.path.join(out_dir, f"decoder_clm_final_step{opt_step}.pt")
        save_checkpoint(ckpt, raw_model, optimizer, opt_step, scaler, metadata=checkpoint_metadata(cfg, None, tgt_tokenizer, raw_model))


def train_mt(cfg):
    # Setup distributed training
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = get_device()
    
    if is_main_process():
        print(f"[mt] rank={rank} world_size={world_size} local_rank={local_rank}")
    
    amp_dtype = get_amp_dtype(cfg["training"].get("amp_dtype", "fp16"))
    out_dir = output_dir(cfg, "mt")
    
    # Only save config from rank 0
    if is_main_process():
        save_run_config(cfg, out_dir)
    
    logger = MetricLogger(out_dir, "mt") if is_main_process() else None

    src_tokenizer, tgt_tokenizer = prepare_tokenizers(cfg)
    train_dataset = ParallelDataset(
        data_path(cfg, "train", cfg["data"]["src_lang"]),
        data_path(cfg, "train", cfg["data"]["tgt_lang"]),
        src_tokenizer,
        model_cfg(cfg, "encoder")["max_seq_len"],
        tgt_tokenizer=tgt_tokenizer,
    )
    valid_dataset = ParallelDataset(
        data_path(cfg, "test", cfg["data"]["src_lang"]),
        data_path(cfg, "test", cfg["data"]["tgt_lang"]),
        src_tokenizer,
        model_cfg(cfg, "encoder")["max_seq_len"],
        tgt_tokenizer=tgt_tokenizer,
    )

    model = build_mt_model(cfg, src_tokenizer, tgt_tokenizer).to(device)
    load_warm_start(cfg, model, device)
    model = maybe_compile(model, cfg, device)
    model = wrap_parallel_model(model, cfg, device)
    
    if is_main_process():
        print(f"[mt] params={count_parameters(unwrap_model(model))/1e6:.2f}M trainable={count_trainable_parameters(unwrap_model(model))/1e6:.2f}M")

    optimizer = make_optimizer(model, cfg, device)
    scaler = make_scaler(device, amp_dtype)
    optimizer.zero_grad(set_to_none=True)

    def tune_step(batch):
        loss = compute_mt_loss(model, batch, src_tokenizer, tgt_tokenizer, device, amp_dtype)
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)

    # Auto-tune only on rank 0
    if is_main_process():
        batch_size = auto_tune_batch_size(lambda bs: make_loader(train_dataset, cfg, bs, True), tune_step, cfg, device)
    else:
        batch_size = cfg["training"]["batch_size"]
    
    # Broadcast batch size to all ranks
    if world_size > 1:
        batch_size_tensor = torch.tensor(batch_size, device=device)
        torch.distributed.broadcast(batch_size_tensor, src=0)
        batch_size = int(batch_size_tensor.item())
    
    loader = make_loader(train_dataset, cfg, batch_size, True)
    train_eval_dataset = fixed_subset(train_dataset, cfg["evaluation"].get("sample_train", 128), cfg["training"]["seed"])
    valid_eval_dataset = fixed_subset(valid_dataset, cfg["evaluation"].get("sample_valid", 128), cfg["training"]["seed"] + 1)
    grad_accum = cfg["training"]["grad_accum_steps"]
    opt_step = 0
    micro_step = 0
    last_loss = 0.0
    model.train()

    while opt_step < cfg["training"]["max_steps"]:
        # Set epoch for DistributedSampler
        if world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(opt_step // cfg["training"]["max_steps"])
        
        for batch in loader:
            if opt_step >= cfg["training"]["max_steps"]:
                break
            lr = set_lr(optimizer, opt_step + 1, cfg)
            loss = compute_mt_loss(model, batch, src_tokenizer, tgt_tokenizer, device, amp_dtype)
            last_loss = float(loss.item())
            scaler.scale(loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue
            opt_step += 1
            train_step_end(model, optimizer, scaler, cfg)
            
            # Reduce loss across ranks
            if world_size > 1:
                loss_dict = reduce_loss({"loss": last_loss})
                last_loss = loss_dict["loss"]
            
            if opt_step % cfg["training"]["log_every"] == 0 and is_main_process():
                logger.log(stage="mt", split="train", step=opt_step, loss=last_loss, lr=lr, batch_size=batch_size)
                print(f"[mt] step={opt_step} loss={last_loss:.4f} lr={lr:.2e} bs={batch_size}")
            
            if opt_step % cfg["training"]["eval_every"] == 0:
                eval_bs = cfg["evaluation"].get("batch_size", 8)
                train_loss = evaluate_loss(
                    model,
                    train_eval_dataset,
                    cfg["evaluation"].get("loss_batch_size", batch_size),
                    cfg,
                    lambda m, b, d, a: compute_mt_loss(m, b, src_tokenizer, tgt_tokenizer, d, a),
                    max_batches=cfg["evaluation"].get("loss_batches", 20),
                )
                valid_loss = evaluate_loss(
                    model,
                    valid_eval_dataset,
                    cfg["evaluation"].get("loss_batch_size", batch_size),
                    cfg,
                    lambda m, b, d, a: compute_mt_loss(m, b, src_tokenizer, tgt_tokenizer, d, a),
                    max_batches=cfg["evaluation"].get("loss_batches", 20),
                )
                train_bleu, train_chrf = evaluate_mt_metrics(model, train_eval_dataset, src_tokenizer, tgt_tokenizer, cfg, eval_bs, 0)
                valid_bleu, valid_chrf = evaluate_mt_metrics(model, valid_eval_dataset, src_tokenizer, tgt_tokenizer, cfg, eval_bs, 0)
                
                if is_main_process():
                    logger.log(stage="mt", split="train", step=opt_step, loss=train_loss, bleu=train_bleu, chrf=train_chrf, lr=lr, batch_size=batch_size)
                    logger.log(stage="mt", split="valid", step=opt_step, loss=valid_loss, bleu=valid_bleu, chrf=valid_chrf, lr=lr, batch_size=batch_size)
                    print(
                        f"[mt:eval] step={opt_step} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
                        f"train_bleu={train_bleu:.2f} valid_bleu={valid_bleu:.2f} valid_chrf={valid_chrf:.2f}"
                    )
            
            if opt_step % cfg["training"]["save_every"] == 0 and is_main_process():
                ckpt = os.path.join(out_dir, f"mt_step{opt_step}.pt")
                raw_model = unwrap_model(model)
                save_checkpoint(ckpt, raw_model, optimizer, opt_step, scaler, metadata=checkpoint_metadata(cfg, src_tokenizer, tgt_tokenizer, raw_model))
    
    if is_main_process():
        raw_model = unwrap_model(model)
        ckpt = os.path.join(out_dir, f"mt_final_step{opt_step}.pt")
        save_checkpoint(ckpt, raw_model, optimizer, opt_step, scaler, metadata=checkpoint_metadata(cfg, src_tokenizer, tgt_tokenizer, raw_model))


def checkpoint_metadata(cfg, src_tokenizer, tgt_tokenizer, model):
    return {
        "seed": cfg["training"]["seed"],
        "src_tokenizer": getattr(src_tokenizer, "model_path", None),
        "tgt_tokenizer": getattr(tgt_tokenizer, "model_path", None),
        "parameters": count_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
    }


def evaluate_mt(cfg):
    device = get_device()
    src_tokenizer, tgt_tokenizer = prepare_tokenizers(cfg)
    dataset = ParallelDataset(
        data_path(cfg, "test", cfg["data"]["src_lang"]),
        data_path(cfg, "test", cfg["data"]["tgt_lang"]),
        src_tokenizer,
        model_cfg(cfg, "encoder")["max_seq_len"],
        tgt_tokenizer=tgt_tokenizer,
    )
    model = build_mt_model(cfg, src_tokenizer, tgt_tokenizer).to(device)
    ckpt = cfg.get("evaluation", {}).get("checkpoint")
    if not ckpt:
        raise ValueError("Set evaluation.checkpoint in the config for --stage eval")
    load_checkpoint(ckpt, model, map_location=device)
    batch_size = cfg["evaluation"].get("batch_size", 8)
    max_samples = cfg["evaluation"].get("final_samples", 0)
    bleu, chrf = evaluate_mt_metrics(model, dataset, src_tokenizer, tgt_tokenizer, cfg, batch_size, max_samples)
    print(f"[eval] BLEU={bleu:.2f} CHRF++={chrf:.2f}")


def load_metric_rows(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def plot_metrics(cfg):
    import matplotlib.pyplot as plt

    root = cfg["training"].get("output_dir", "checkpoints")
    paths = [
        os.path.join(root, "mt", "mt_metrics.jsonl"),
        cfg.get("evaluation", {}).get("metrics_path", ""),
    ]
    rows = load_metric_rows(paths)
    if not rows:
        raise ValueError("No metric rows found for plotting")

    plot_dir = os.path.join(root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for metric, filename, ylabel in [
        ("loss", "loss.png", "Cross-entropy loss"),
        ("bleu", "bleu_100.png", "BLEU-100"),
        ("chrf", "chrfpp_100.png", "CHRF++-100"),
    ]:
        plt.figure(figsize=(8, 5))
        plotted = False
        for split in ["train", "valid"]:
            xs = [r["step"] for r in rows if r.get("split") == split and r.get(metric) not in (None, "")]
            ys = [float(r[metric]) for r in rows if r.get("split") == split and r.get(metric) not in (None, "")]
            if xs:
                plt.plot(xs, ys, label=split)
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Optimizer step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(plot_dir, filename)
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[plot] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mlm", "clm", "mt", "eval", "plot"])
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (auto-set by launcher)")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        repo_config_path = os.path.join(repo_root, config_path)
        if os.path.exists(repo_config_path):
            config_path = repo_config_path

    os.chdir(repo_root)
    cfg = load_config(config_path)
    cfg.setdefault("evaluation", {})

    # If the user launched with torchrun, initialize DDP. Otherwise the model
    # stage code will fall back to DataParallel when multiple GPUs are visible.
    if args.local_rank != -1 or "RANK" in os.environ:
        setup_distributed(cfg.get("distributed", {}).get("backend"))

    set_seed(cfg["training"]["seed"])

    try:
        if args.stage == "mlm":
            train_mlm(cfg)
        elif args.stage == "clm":
            train_clm(cfg)
        elif args.stage == "mt":
            train_mt(cfg)
        elif args.stage == "eval":
            evaluate_mt(cfg)
        else:
            plot_metrics(cfg)
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()
