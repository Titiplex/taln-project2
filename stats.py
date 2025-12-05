from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Sequence, Dict, Any
import csv

from src.dataset import collect_sentences as collect
import src.config as cfg


def compute_sentence_stats(sentences: Sequence[str]) -> Dict[str, Any]:
    """
    Computes :
      - sentence count
      - token average length (split())
      - length variance
      - vocab size (lowercase)
    """
    n = len(sentences)
    if n == 0:
        return {
            "n_sentences": 0,
            "mean_len": 0.0,
            "var_len": 0.0,
            "vocab_size": 0,
        }

    lengths = []
    vocab = set()

    for s in sentences:
        # simple tokens
        toks = [t for t in s.split() if t.strip()]
        lengths.append(len(toks))
        for t in toks:
            vocab.add(t.lower())

    mean_len = sum(lengths) / n
    var_len = sum((l - mean_len) ** 2 for l in lengths) / n

    return {
        "n_sentences": n,
        "mean_len": mean_len,
        "var_len": var_len,
        "vocab_size": len(vocab),
    }


def print_sentence_stats(stats: Dict[str, Any]) -> None:
    """
    Clean display
    """
    print("=== Corpus statstics ===")
    print(f"Sentence number        : {stats['n_sentences']}")
    print(f"Average length (tokens): {stats['mean_len']:.2f}")
    print(f"Length variance        : {stats['var_len']:.2f}")
    print(f"Vocabulary size        : {stats['vocab_size']}")
    print()


def compute_entity_stats(path: Path, top_k: int = 20) -> Dict[str, Any]:
    """
    Reads entities.csv :
      - total nb of lines
      - label distribution (PERSON, ORG, GPE, etc)
      - top_k most frequent normalized forms
    """
    label_counts = Counter()
    norm_counts = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        required = {"label", "normalized"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain at least the colomns {required}")

        for row in reader:
            total += 1
            label = row["label"].strip()
            if label:
                label_counts[label] += 1

            norm = row.get("normalized", "").strip().lower()
            if norm:
                norm_counts[norm] += 1

    return {
        "total": total,
        "label_counts": label_counts,
        "top_normalized": norm_counts.most_common(top_k),
    }


def print_entity_stats(stats: Dict[str, Any]) -> None:
    print("=== Named entities ===")
    print(f"Total number of extracted entities : {stats['total']}")
    print("\nFor each label :")
    for label, cnt in stats["label_counts"].most_common():
        print(f"  {label:8s} : {cnt}")

    print("\nTop normalized forms :")
    for norm, cnt in stats["top_normalized"]:
        print(f"  {norm!r:30s} -> {cnt}")
    print()

def compute_acronym_stats(path: Path, top_k: int = 20) -> Dict[str, Any]:
    """
    Reads acronyms.csv :
      - total number of lines
      - distinct acronym number
      - Acronym distribution
    """
    total = 0
    acr_counts = Counter()

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if "acronym" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a column 'acronym'")

        for row in reader:
            total += 1
            acr = row["acronym"].strip()
            if acr:
                acr_counts[acr] += 1

    return {
        "total": total,
        "n_unique_acronyms": len(acr_counts),
        "top_acronyms": acr_counts.most_common(top_k),
    }


def print_acronym_stats(stats: Dict[str, Any]) -> None:
    print("=== Acronyms ===")
    print(f"Extracted lines           : {stats['total']}")
    print(f"Distinct acronyms         : {stats['n_unique_acronyms']}")
    print("\nTop acronyms :")
    for acr, cnt in stats["top_acronyms"]:
        print(f"  {acr:15s} -> {cnt}")
    print()


def compute_is_a_stats(path: Path, top_k: int = 20) -> Dict[str, Any]:
    """
    Reads is_a_relations.csv :
      - total number of relations
      - top_k most frequent hyponyms
      - top_k most frequent hypernyms
    """
    total = 0
    hypo_counts = Counter()
    hyper_counts = Counter()

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        required = {"hyponym", "hypernym"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain the columns {required}")

        for row in reader:
            total += 1
            hypo = row["hyponym"].strip()
            hyper = row["hypernym"].strip()
            if hypo:
                hypo_counts[hypo] += 1
            if hyper:
                hyper_counts[hyper] += 1

    return {
        "total": total,
        "top_hyponyms": hypo_counts.most_common(top_k),
        "top_hypernyms": hyper_counts.most_common(top_k),
    }


def print_is_a_stats(stats: Dict[str, Any]) -> None:
    print("=== Relations IS_A ===")
    print(f"Total number of relations : {stats['total']}")
    print("\nTop hyponyms :")
    for h, cnt in stats["top_hyponyms"]:
        print(f"  {h!r:40s} -> {cnt}")

    print("\nTop hypernyms :")
    for h, cnt in stats["top_hypernyms"]:
        print(f"  {h!r:40s} -> {cnt}")
    print()


def main() -> None:
    print(">>> Computing stats on corpus...")
    sentences = collect(cfg.N_SENTENCES)
    s_stats = compute_sentence_stats(sentences)
    print_sentence_stats(s_stats)

    outputs_dir = Path("outputs")

    ent_path = outputs_dir / "entities.csv"
    acr_path = outputs_dir / "acronyms.csv"
    is_a_path = outputs_dir / "is_a_relations.csv"

    if ent_path.exists():
        e_stats = compute_entity_stats(ent_path)
        print_entity_stats(e_stats)
    else:
        print(f"[WARN] {ent_path} not found, skip entities.")

    if acr_path.exists():
        a_stats = compute_acronym_stats(acr_path)
        print_acronym_stats(a_stats)
    else:
        print(f"[WARN] {acr_path} not found, skip acronyms.")

    if is_a_path.exists():
        t_stats = compute_is_a_stats(is_a_path)
        print_is_a_stats(t_stats)
    else:
        print(f"[WARN] {is_a_path} not found, skip IS_A.")


if __name__ == "__main__":
    main()
