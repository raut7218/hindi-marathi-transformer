import os
import torch


def save_checkpoint(path, model, optimizer, step, scaler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)
