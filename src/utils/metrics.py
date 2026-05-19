def compute_bleu_chrf(preds, refs):
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise ImportError("Install sacrebleu to compute BLEU and CHRF++: pip install -r requirements.txt") from exc

    bleu = BLEU()
    chrf = CHRF(word_order=2)
    bleu_score = bleu.corpus_score(preds, [refs]).score
    chrf_score = chrf.corpus_score(preds, [refs]).score
    return bleu_score, chrf_score
