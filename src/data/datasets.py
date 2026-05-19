import random
import torch
from torch.utils.data import Dataset


def read_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


class MonolingualDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = read_lines(path)
        self.encoded = [self._encode(line) for line in self.lines]

    def _encode(self, line):
        ids = self.tokenizer.encode(line, add_bos=True, add_eos=True)
        return ids[: self.max_len]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


class ParallelDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, max_len, tgt_tokenizer=None):
        self.src_tokenizer = tokenizer
        self.tgt_tokenizer = tgt_tokenizer or tokenizer
        self.max_len = max_len
        self.src_lines = read_lines(src_path)
        self.tgt_lines = read_lines(tgt_path)
        assert len(self.src_lines) == len(self.tgt_lines)
        self.src_encoded = [self._encode(s, self.src_tokenizer) for s in self.src_lines]
        self.tgt_encoded = [self._encode(t, self.tgt_tokenizer) for t in self.tgt_lines]

    def _encode(self, line, tokenizer):
        ids = tokenizer.encode(line, add_bos=True, add_eos=True)
        return ids[: self.max_len]

    def __len__(self):
        return len(self.src_encoded)

    def __getitem__(self, idx):
        return self.src_encoded[idx], self.tgt_encoded[idx]


def pad_sequences(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return batch


def make_mlm_batch(batch, tokenizer, mask_prob=0.15):
    pad_id = tokenizer.pad_id
    mask_id = tokenizer.mask_id
    vocab_size = tokenizer.sp.get_piece_size()
    input_ids = pad_sequences(batch, pad_id)
    labels = input_ids.clone()

    special_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}

    for i in range(input_ids.size(0)):
        for j in range(input_ids.size(1)):
            token_id = int(input_ids[i, j].item())
            if token_id in special_ids:
                labels[i, j] = -100
                continue
            if random.random() < mask_prob:
                r = random.random()
                if r < 0.8:
                    input_ids[i, j] = mask_id
                elif r < 0.9:
                    input_ids[i, j] = random.randint(0, vocab_size - 1)
                else:
                    input_ids[i, j] = token_id
            else:
                labels[i, j] = -100

    attention_mask = input_ids.ne(pad_id).long()
    return input_ids, attention_mask, labels


def make_clm_batch(batch, tokenizer):
    pad_id = tokenizer.pad_id
    input_ids = pad_sequences(batch, pad_id)
    labels = input_ids.clone()

    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels[input_ids == pad_id] = -100

    attention_mask = input_ids.ne(pad_id).long()
    return input_ids, attention_mask, labels


def make_mt_batch(batch, src_tokenizer, tgt_tokenizer=None):
    tgt_tokenizer = tgt_tokenizer or src_tokenizer
    src_pad_id = src_tokenizer.pad_id
    tgt_pad_id = tgt_tokenizer.pad_id
    src, tgt = zip(*batch)
    src_ids = pad_sequences(src, src_pad_id)
    tgt_ids = pad_sequences(tgt, tgt_pad_id)

    decoder_in = tgt_ids[:, :-1]
    labels = tgt_ids[:, 1:].clone()
    labels[labels == tgt_pad_id] = -100

    src_mask = src_ids.ne(src_pad_id).long()
    tgt_mask = decoder_in.ne(tgt_pad_id).long()
    return src_ids, src_mask, decoder_in, tgt_mask, labels
