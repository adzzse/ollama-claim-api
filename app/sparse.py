import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

_TOKENIZER = None
MODEL_ID = "naver/splade-v3-distilbert"


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    logger.info("Loaded tokenizer %s (vocab_size=%s)", MODEL_ID, _TOKENIZER.vocab_size)
    return _TOKENIZER


def encode_sparse(text: str) -> dict:
    tokenizer = _get_tokenizer()
    ids = tokenizer(text)["input_ids"]

    # Exclude special tokens ([CLS]=101, [SEP]=102, [PAD]=0)
    counts: Counter[int] = Counter()
    for tid in ids:
        if tid not in (101, 102, 0):
            counts[tid] += 1

    indices: list[int] = []
    values: list[float] = []
    for tid, count in counts.items():
        indices.append(tid)
        values.append(math.log(1.0 + count))

    return {"indices": indices, "values": values}
