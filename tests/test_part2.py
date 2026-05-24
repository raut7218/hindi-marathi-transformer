import json
import sys
from pathlib import Path

import torch
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import scripts.train as train_module
from scripts.train import plot_metrics
from src.data.datasets import make_mlm_batch
from src.models.transformer import DecoderModel, EncoderModel, Seq2SeqModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint
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
    assert encoder(src_ids, src_mask, return_logits=True).shape == (2, 7, encoder_cfg["vocab_size"])
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


def test_prepare_tokenizers_distributed_uses_one_matching_barrier(monkeypatch):
    cfg = {
        "data": {"data_dir": "data", "src_lang": "hi", "tgt_lang": "mr"},
        "tokenizers": {
            "shared": False,
            "src": {"model_prefix": "data/spm_hi", "vocab_size": 100, "character_coverage": 0.9995},
            "tgt": {"model_prefix": "data/spm_mr", "vocab_size": 100, "character_coverage": 0.9995},
        },
    }

    class FakeTokenizer:
        def __init__(self, model_path):
            self.model_path = model_path

    for main_process in (True, False):
        calls = {"sync": 0, "train": 0}

        def fake_train(input_paths, model_prefix, *args):
            calls["train"] += 1
            return model_prefix + ".model"

        monkeypatch.setattr(train_module, "get_world_size", lambda: 2)
        monkeypatch.setattr(train_module, "is_main_process", lambda main=main_process: main)
        monkeypatch.setattr(train_module, "synchronize", lambda: calls.__setitem__("sync", calls["sync"] + 1))
        monkeypatch.setattr(train_module, "train_sentencepiece_from_files", fake_train)
        monkeypatch.setattr(train_module, "SentencePieceTokenizer", FakeTokenizer)

        src_tokenizer, tgt_tokenizer = train_module.prepare_tokenizers(cfg)

        assert calls["sync"] == 1
        assert calls["train"] == (2 if main_process else 0)
        assert src_tokenizer.model_path == "data/spm_hi.model"
        assert tgt_tokenizer.model_path == "data/spm_mr.model"


def test_select_batch_size_ddp_skips_auto_tune(monkeypatch):
    cfg = {"training": {"batch_size": 8, "auto_batch_size": True}}

    def fail_auto_tune(*args, **kwargs):
        raise AssertionError("DDP should not run rank-local auto tuning")

    monkeypatch.setattr(train_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(train_module, "is_main_process", lambda: True)
    monkeypatch.setattr(train_module, "auto_tune_batch_size", fail_auto_tune)

    batch_size = train_module.select_batch_size(
        lambda bs: None,
        lambda batch: None,
        cfg,
        torch.device("cpu"),
    )

    assert batch_size == 8


def test_compute_mlm_loss_uses_wrapped_forward(monkeypatch):
    class FakeTokenizer:
        pad_id = 0

    class FakeInner:
        def forward_mlm(self, *args, **kwargs):
            raise AssertionError("MLM loss should not bypass the wrapper")

    class FakeWrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = FakeInner()
            self.calls = []

        def forward(self, input_ids, attn_mask=None, return_logits=False):
            self.calls.append((input_ids, attn_mask, return_logits))
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), 5)
            logits[..., 4] = 1.0
            return logits

    monkeypatch.setattr(
        train_module,
        "make_mlm_batch",
        lambda batch, tokenizer: (
            torch.tensor([[2, 4, 3]]),
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[-100, 4, -100]]),
        ),
    )

    model = FakeWrapped()
    loss = train_module.compute_mlm_loss(model, [[2, 4, 3]], FakeTokenizer(), torch.device("cpu"), torch.float32)

    assert loss.item() > 0
    assert model.calls
    assert model.calls[0][2] is True


def test_vectorized_mlm_masking_preserves_special_tokens():
    class FakeSP:
        def get_piece_size(self):
            return 20

    class FakeTokenizer:
        pad_id = 0
        bos_id = 2
        eos_id = 3
        mask_id = 4
        sp = FakeSP()

    torch.manual_seed(0)
    input_ids, attention_mask, labels = make_mlm_batch(
        [[2, 5, 6, 3], [2, 7, 3]],
        FakeTokenizer(),
        mask_prob=1.0,
    )

    assert input_ids.shape == labels.shape == attention_mask.shape
    assert labels[0, 0].item() == -100
    assert labels[0, 3].item() == -100
    assert labels[1, 3].item() == -100
    assert labels[0, 1].item() == 5
    assert labels[0, 2].item() == 6
    assert labels[1, 1].item() == 7
    assert attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_backward_context_uses_no_sync_for_non_final_ddp_microstep(monkeypatch):
    class FakeDDP:
        def __init__(self):
            self.used_no_sync = False

        def no_sync(self):
            model = self

            class Context:
                def __enter__(self):
                    model.used_no_sync = True

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Context()

    monkeypatch.setattr(train_module.torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    model = FakeDDP()
    with train_module.backward_context(model, sync_gradients=False):
        pass

    assert model.used_no_sync is True


def test_stage_lr_and_mt_freeze_lr_scaling():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW([
        {"params": [model.weight], "lr_scale": 1.0},
        {"params": [model.bias], "lr_scale": 0.3},
    ])
    cfg = {
        "training": {
            "lr": 3.0e-4,
            "stage_lrs": {"mt": 1.0e-4},
            "warmup_steps": 10,
            "max_steps": 100,
        }
    }

    lr = train_module.set_lr(optimizer, 10, cfg, "mt", pretrained_frozen=True)

    assert lr == 1.0e-4
    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert optimizer.param_groups[1]["lr"] == 0.0


def test_checkpoint_save_is_atomic_and_removes_tmp(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, step=3)

    assert path.exists()
    assert not (tmp_path / "model.pt.tmp").exists()
    payload = torch.load(path, map_location="cpu")
    assert payload["step"] == 3
    assert "model" in payload


def test_model_only_checkpoint_excludes_optimizer_and_scaler(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "model_only.pt"

    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        step=5,
        scaler=scaler,
        save_optimizer=False,
        save_scaler=False,
    )

    payload = torch.load(path, map_location="cpu")
    assert "optimizer" not in payload
    assert "scaler" not in payload
    assert payload["step"] == 5


def test_load_checkpoint_supports_full_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "full.pt"
    save_checkpoint(path, model, optimizer=optimizer, step=7)

    loaded_model = torch.nn.Linear(2, 2)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters())
    step = load_checkpoint(path, loaded_model, optimizer=loaded_optimizer, map_location="cpu")

    assert step == 7


def test_colab_config_is_single_gpu_and_model_only():
    cfg = yaml.safe_load((repo_root / "configs" / "colab_t4.yaml").read_text(encoding="utf-8"))

    assert cfg["distributed"]["enabled"] is False
    assert cfg["training"]["output_dir"] == "checkpoints_colab"
    assert cfg["training"]["batch_size"] == 10
    assert cfg["training"]["grad_accum_steps"] == 6
    assert cfg["training"]["max_steps"] == 1000
    assert cfg["training"]["warmup_steps"] == 100
    assert cfg["training"]["save_every"] == 0
    assert cfg["training"]["save_optimizer_state"] is False
    assert cfg["mt"]["encoder_checkpoint"].startswith("checkpoints_colab/")
    assert cfg["mt"]["encoder_checkpoint"].endswith("step1000.pt")
