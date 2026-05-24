import torch


def _unwrap_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return getattr(model, "_orig_mod", model)


def greedy_decode(model, src_ids, src_mask, bos_id, eos_id, max_len, device):
    model = _unwrap_model(model)
    model.eval()
    with torch.no_grad():
        enc_out = model.encoder(src_ids, src_mask)
        enc_mask = (1.0 - src_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        ys = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_mask = torch.ones_like(ys, dtype=torch.long)
            logits = model.decoder(ys, tgt_mask=tgt_mask, enc_mask=enc_mask, enc_out=enc_out)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            if torch.all(finished):
                break
    return ys
