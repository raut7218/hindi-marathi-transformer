from sacrebleu.metrics import BLEU, CHRF


def compute_bleu_chrf(preds, refs):
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    bleu_score = bleu.corpus_score(preds, [refs]).score
    chrf_score = chrf.corpus_score(preds, [refs]).score
    return bleu_score, chrf_score
