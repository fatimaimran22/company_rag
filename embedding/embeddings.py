"""Turn text into vectors with a Sentence Transformers model, and compare vectors.

What all-MiniLM-L6-v2 does with one string (you can see the 3 stages by
printing the model object):

  1. Transformer: split the text into word-piece tokens and produce one
     384-number vector PER TOKEN. Only the first 256 tokens are read;
     anything after that is silently ignored ("truncation").
  2. Pooling: average all token vectors into ONE 384-number vector (mean pooling).
  3. Normalize: scale that vector to length 1.

Because of step 3, cosine similarity between two of our vectors is simply
their dot product.
"""

import os
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()  # hide "sequence longer than max length" noise; we report it ourselves
hf_logging.disable_progress_bar()


def load_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name, device="cpu")


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Return a (len(texts), dim) float32 matrix, one row per text."""
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)


@dataclass
class TokenInfo:
    token_count: int  # tokens incl. the [CLS]/[SEP] markers the model adds
    max_seq_length: int  # how many tokens the model reads
    truncated: bool
    embedded_chars: int  # how many characters of the text the model actually read


def token_info(model: SentenceTransformer, text: str) -> TokenInfo:
    """How much of `text` does the model actually read?"""
    enc = model.tokenizer(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    count, limit = len(enc["input_ids"]), model.max_seq_length
    if count <= limit:
        return TokenInfo(count, limit, False, len(text))
    # Kept tokens: [CLS] + (limit - 2) text tokens + [SEP]. The last kept text token ends here:
    last_kept_end = enc["offset_mapping"][limit - 2][1]
    return TokenInfo(count, limit, True, last_kept_end)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(angle) = (a . b) / (|a| * |b|). 1.0 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
