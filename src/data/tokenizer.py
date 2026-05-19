import os
import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def unk_id(self):
        return self.sp.unk_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def mask_id(self):
        return self.sp.piece_to_id("[MASK]")

    def encode(self, text, add_bos=True, add_eos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)


def train_sentencepiece_from_files(
    input_paths,
    model_prefix,
    vocab_size,
    character_coverage,
    model_type="unigram",
    input_sentence_size=0,
    max_sentence_length=4192,
):
    model_path = model_prefix + ".model"
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    corpus_path = model_prefix + "_corpus.txt"

    if not os.path.exists(corpus_path):
        with open(corpus_path, "w", encoding="utf-8") as out_f:
            for path in input_paths:
                with open(path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols="[MASK]",
        hard_vocab_limit=False,
    )
    return model_path


def train_sentencepiece(data_dir, model_prefix, vocab_size, character_coverage):
    src_path = os.path.join(data_dir, "train.hi")
    tgt_path = os.path.join(data_dir, "train.mr")
    return train_sentencepiece_from_files(
        [src_path, tgt_path],
        model_prefix,
        vocab_size,
        character_coverage,
    )
