import json
from pathlib import Path

import torch

from scripts.train import plot_metrics
from src.models.transformer import DecoderModel, EncoderModel, Seq2SeqModel
from src.utils.params import count_parameters


def test_part2_parameter_counts_on_meta_device():
    with torch.device("meta"):
        encoder = EncoderModel(
            vocab_size=45000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            ffn_dim=3072,
            dropout=0.1,
            max_seq_len=128,
        )
        decoder = DecoderModel(
            vocab_size=50257,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            ffn_dim=3584,
            dropout=0.1,
            max_seq_len=128,
        )

    assert count_parameters(encoder) == 110_076_672
    assert count_parameters(decoder) == 123_551_232


def test_encoder_decoder_and_mt_shapes_with_separate_vocabularies():
    encoder_cfg = {
        "vocab_size": 37,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "max_seq_len": 16,
    }
    decoder_cfg = {
        "vocab_size": 43,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_dim": 80,
        "dropout": 0.0,
        "max_seq_len": 16,
    }
    encoder = EncoderModel(**encoder_cfg)
    decoder = DecoderModel(**decoder_cfg)
    mt_model = Seq2SeqModel(encoder_config=encoder_cfg, decoder_config=decoder_cfg)

    src_ids = torch.randint(0, encoder_cfg["vocab_size"], (2, 7))
    tgt_ids = torch.randint(0, decoder_cfg["vocab_size"], (2, 6))
    src_mask = torch.ones_like(src_ids)
    tgt_mask = torch.ones_like(tgt_ids)

    assert encoder.forward_mlm(src_ids, src_mask).shape == (2, 7, encoder_cfg["vocab_size"])
    assert decoder(tgt_ids, tgt_mask=tgt_mask).shape == (2, 6, decoder_cfg["vocab_size"])
    assert mt_model(src_ids, src_mask, tgt_ids, tgt_mask=tgt_mask).shape == (2, 6, decoder_cfg["vocab_size"])


def test_plot_generation_from_fake_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {"stage": "mt", "split": "train", "step": 1, "loss": 4.0, "bleu": 1.0, "chrf": 10.0},
        {"stage": "mt", "split": "valid", "step": 1, "loss": 4.5, "bleu": 0.5, "chrf": 8.0},
        {"stage": "mt", "split": "train", "step": 2, "loss": 3.5, "bleu": 2.0, "chrf": 12.0},
        {"stage": "mt", "split": "valid", "step": 2, "loss": 4.0, "bleu": 1.0, "chrf": 9.0},
    ]
    metrics_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    cfg = {
        "training": {"output_dir": str(tmp_path / "out")},
        "evaluation": {"metrics_path": str(metrics_path)},
    }

    plot_metrics(cfg)

    assert (tmp_path / "out" / "plots" / "loss.png").exists()
    assert (tmp_path / "out" / "plots" / "bleu_100.png").exists()
    assert (tmp_path / "out" / "plots" / "chrfpp_100.png").exists()
