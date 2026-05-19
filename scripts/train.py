import argparse
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.schedule import get_lr
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics import compute_bleu_chrf
from src.utils.decoding import greedy_decode
from src.data.tokenizer import train_sentencepiece, SentencePieceTokenizer
from src.data.datasets import (
    MonolingualDataset,
    ParallelDataset,
    make_mlm_batch,
    make_clm_batch,
    make_mt_batch,
)
from src.models.transformer import EncoderModel, DecoderModel, Seq2SeqModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_tune_batch_size(make_loader, test_step, start_bs):
    bs = start_bs
    while bs >= 1:
        try:
            loader = make_loader(bs)
            batch = next(iter(loader))
            test_step(batch)
            return bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                bs = bs // 2
            else:
                raise e
    raise RuntimeError("Unable to find a working batch size")


def train_mlm(cfg):
    device = get_device()
    data_dir = cfg["data"]["data_dir"]

    model_path = train_sentencepiece(
        data_dir,
        cfg["tokenizer"]["model_prefix"],
        cfg["tokenizer"]["vocab_size"],
        cfg["tokenizer"]["character_coverage"],
    )
    tokenizer = SentencePieceTokenizer(model_path)

    train_path = os.path.join(data_dir, f"train.{cfg['data']['src_lang']}")
    dataset = MonolingualDataset(train_path, tokenizer, cfg["model"]["max_seq_len"])

    def make_loader(bs):
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=cfg["training"]["num_workers"])

    model = EncoderModel(
        vocab_size=tokenizer.sp.get_piece_size(),
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        ffn_mult=cfg["model"]["ffn_mult"],
        dropout=cfg["model"]["dropout"],
        max_seq_len=cfg["model"]["max_seq_len"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["training"]["amp_dtype"] == "fp16"))

    def test_step(batch):
        input_ids, attn_mask, labels = make_mlm_batch(batch, tokenizer)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model.forward_mlm(input_ids, attn_mask=attn_mask)
            _ = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    bs = cfg["training"]["batch_size"]
    if cfg["training"].get("auto_batch_size", False):
        bs = auto_tune_batch_size(make_loader, test_step, bs)

    loader = make_loader(bs)
    step = 0
    model.train()

    for epoch in range(10**6):
        for batch in loader:
            step += 1
            lr = get_lr(step, cfg["training"]["lr"], cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids, attn_mask, labels = make_mlm_batch(batch, tokenizer)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(dtype=getattr(torch, cfg["training"]["amp_dtype"]), enabled=(device.type == "cuda")):
                logits = model.forward_mlm(input_ids, attn_mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / cfg["training"]["grad_accum_steps"]

            scaler.scale(loss).backward()
            if step % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg["training"]["log_every"] == 0:
                print(f"[mlm] step={step} loss={loss.item() * cfg['training']['grad_accum_steps']:.4f}")

            if step % cfg["training"]["save_every"] == 0:
                ckpt = os.path.join(cfg["training"]["output_dir"], f"encoder_mlm_step{step}.pt")
                save_checkpoint(ckpt, model, optimizer, step, scaler)

            if step >= cfg["training"]["max_steps"]:
                return


def train_clm(cfg):
    device = get_device()
    data_dir = cfg["data"]["data_dir"]

    model_path = train_sentencepiece(
        data_dir,
        cfg["tokenizer"]["model_prefix"],
        cfg["tokenizer"]["vocab_size"],
        cfg["tokenizer"]["character_coverage"],
    )
    tokenizer = SentencePieceTokenizer(model_path)

    train_path = os.path.join(data_dir, f"train.{cfg['data']['tgt_lang']}")
    dataset = MonolingualDataset(train_path, tokenizer, cfg["model"]["max_seq_len"])

    def make_loader(bs):
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=cfg["training"]["num_workers"])

    model = DecoderModel(
        vocab_size=tokenizer.sp.get_piece_size(),
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        ffn_mult=cfg["model"]["ffn_mult"],
        dropout=cfg["model"]["dropout"],
        max_seq_len=cfg["model"]["max_seq_len"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["training"]["amp_dtype"] == "fp16"))

    def test_step(batch):
        input_ids, attn_mask, labels = make_clm_batch(batch, tokenizer)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(input_ids, tgt_mask=attn_mask)
            _ = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    bs = cfg["training"]["batch_size"]
    if cfg["training"].get("auto_batch_size", False):
        bs = auto_tune_batch_size(make_loader, test_step, bs)

    loader = make_loader(bs)
    step = 0
    model.train()

    for epoch in range(10**6):
        for batch in loader:
            step += 1
            lr = get_lr(step, cfg["training"]["lr"], cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids, attn_mask, labels = make_clm_batch(batch, tokenizer)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(dtype=getattr(torch, cfg["training"]["amp_dtype"]), enabled=(device.type == "cuda")):
                logits = model(input_ids, tgt_mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / cfg["training"]["grad_accum_steps"]

            scaler.scale(loss).backward()
            if step % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg["training"]["log_every"] == 0:
                print(f"[clm] step={step} loss={loss.item() * cfg['training']['grad_accum_steps']:.4f}")

            if step % cfg["training"]["save_every"] == 0:
                ckpt = os.path.join(cfg["training"]["output_dir"], f"decoder_clm_step{step}.pt")
                save_checkpoint(ckpt, model, optimizer, step, scaler)

            if step >= cfg["training"]["max_steps"]:
                return


def train_mt(cfg):
    device = get_device()
    data_dir = cfg["data"]["data_dir"]

    model_path = train_sentencepiece(
        data_dir,
        cfg["tokenizer"]["model_prefix"],
        cfg["tokenizer"]["vocab_size"],
        cfg["tokenizer"]["character_coverage"],
    )
    tokenizer = SentencePieceTokenizer(model_path)

    src_path = os.path.join(data_dir, f"train.{cfg['data']['src_lang']}")
    tgt_path = os.path.join(data_dir, f"train.{cfg['data']['tgt_lang']}")
    dataset = ParallelDataset(src_path, tgt_path, tokenizer, cfg["model"]["max_seq_len"])

    def make_loader(bs):
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=cfg["training"]["num_workers"])

    model = Seq2SeqModel(
        vocab_size=tokenizer.sp.get_piece_size(),
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        ffn_mult=cfg["model"]["ffn_mult"],
        dropout=cfg["model"]["dropout"],
        max_seq_len=cfg["model"]["max_seq_len"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["training"]["amp_dtype"] == "fp16"))

    def test_step(batch):
        src_ids, src_mask, tgt_in, tgt_mask, labels = make_mt_batch(batch, tokenizer)
        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)
        tgt_in = tgt_in.to(device)
        tgt_mask = tgt_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(src_ids, src_mask, tgt_in, tgt_mask=tgt_mask)
            _ = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    bs = cfg["training"]["batch_size"]
    if cfg["training"].get("auto_batch_size", False):
        bs = auto_tune_batch_size(make_loader, test_step, bs)

    loader = make_loader(bs)
    step = 0
    model.train()

    for epoch in range(10**6):
        for batch in loader:
            step += 1
            lr = get_lr(step, cfg["training"]["lr"], cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            src_ids, src_mask, tgt_in, tgt_mask, labels = make_mt_batch(batch, tokenizer)
            src_ids = src_ids.to(device)
            src_mask = src_mask.to(device)
            tgt_in = tgt_in.to(device)
            tgt_mask = tgt_mask.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(dtype=getattr(torch, cfg["training"]["amp_dtype"]), enabled=(device.type == "cuda")):
                logits = model(src_ids, src_mask, tgt_in, tgt_mask=tgt_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / cfg["training"]["grad_accum_steps"]

            scaler.scale(loss).backward()
            if step % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg["training"]["log_every"] == 0:
                print(f"[mt] step={step} loss={loss.item() * cfg['training']['grad_accum_steps']:.4f}")

            if step % cfg["training"]["eval_every"] == 0:
                evaluate_mt(cfg, model, tokenizer, device)

            if step % cfg["training"]["save_every"] == 0:
                ckpt = os.path.join(cfg["training"]["output_dir"], f"mt_step{step}.pt")
                save_checkpoint(ckpt, model, optimizer, step, scaler)

            if step >= cfg["training"]["max_steps"]:
                return


def evaluate_mt(cfg, model, tokenizer, device):
    data_dir = cfg["data"]["data_dir"]
    src_path = os.path.join(data_dir, f"test.{cfg['data']['src_lang']}")
    tgt_path = os.path.join(data_dir, f"test.{cfg['data']['tgt_lang']}")
    dataset = ParallelDataset(src_path, tgt_path, tokenizer, cfg["model"]["max_seq_len"])
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    preds = []
    refs = []
    for batch in loader:
        src_ids, src_mask, _, _, labels = make_mt_batch(batch, tokenizer)
        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)
        out_ids = greedy_decode(
            model,
            src_ids,
            src_mask,
            tokenizer.bos_id,
            tokenizer.eos_id,
            cfg["model"]["max_seq_len"],
            device,
        )
        out_ids = out_ids[:, 1:].tolist()
        for ids in out_ids:
            if tokenizer.eos_id in ids:
                ids = ids[: ids.index(tokenizer.eos_id)]
            preds.append(tokenizer.decode(ids))
        for lbl in labels.tolist():
            lbl = [t for t in lbl if t != -100 and t != tokenizer.pad_id]
            refs.append(tokenizer.decode(lbl))

    bleu, chrf = compute_bleu_chrf(preds, refs)
    print(f"[eval] BLEU={bleu:.2f} CHRF={chrf:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["mlm", "clm", "mt"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    if args.stage == "mlm":
        train_mlm(cfg)
    elif args.stage == "clm":
        train_clm(cfg)
    else:
        train_mt(cfg)


if __name__ == "__main__":
    main()
