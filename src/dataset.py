from typing import List
from datasets import load_dataset
import spacy

from .config import DATASET_NAME, DATASET_CONFIG, SPACY_MODEL


def build_nlp(model: str = SPACY_MODEL):
    """Loads spacy model"""
    return spacy.load(model)


def collect_sentences(max_sentences: int = 50_000, nlp=None) -> List[str]:
    """
    collects sentences from the dataset
    returns list of sentences (string)
    """
    if nlp is None:
        nlp = build_nlp()

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    texts = ds["train"]["text"]

    sentences: List[str] = []
    for doc in nlp.pipe(texts, batch_size=1000):
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            sentences.append(s)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences