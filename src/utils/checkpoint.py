import torch
from pathlib import Path


def save_checkpoint(
    path,
    model,
    optimizer=None,
    step=0,
    scaler=None,
    metadata=None,
    save_optimizer=True,
    save_scaler=True,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    payload = {
        "model": model.state_dict(),
        "step": step,
    }
    if save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if save_scaler and scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if metadata is not None:
        payload["metadata"] = metadata
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)
