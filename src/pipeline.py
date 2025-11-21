from pathlib import Path

from .config import N_SENTENCES, DATA_DIR
from .dataset import build_nlp, collect_sentences
from .io_utils import write_csv
from .extractors.entities import extract_named_entities
from .extractors.acronyms import extract_acronyms
from .extractors.taxonomy import extract_is_a


def run_extraction(max_sentences: int = N_SENTENCES, out_dir: Path | None = None) -> None:
    """
    collect sentences, launch extractors, write csv files
    """
    if out_dir is None:
        out_dir = DATA_DIR

    print("Loading spaCy model...")
    nlp = build_nlp()

    print(f"Collecting up to {max_sentences} sentences from the dataset...")
    sentences = collect_sentences(max_sentences=max_sentences, nlp=nlp)
    print(f"Collected {len(sentences)} sentences.")

    print("Extracting named entities...")
    ent_rows = extract_named_entities(nlp, sentences)
    write_csv(
        out_dir / "entities.csv",
        ["sent_id", "entity", "label", "normalized", "sentence"],
        ent_rows,
    )
    print(f"Wrote {len(ent_rows)} entity rows.")

    print("Extracting acronyms...")
    acr_rows = extract_acronyms(nlp, sentences)
    write_csv(
        out_dir / "acronyms.csv",
        ["sent_id", "acronym", "long_form", "sentence"],
        acr_rows,
    )
    print(f"Wrote {len(acr_rows)} acronym rows.")

    print("Extracting IS_A relations...")
    tax_rows = extract_is_a(nlp, sentences)
    write_csv(
        out_dir / "is_a_relations.csv",
        ["sent_id", "hyponym", "relation", "hypernym", "sentence"],
        tax_rows,
    )
    print(f"Wrote {len(tax_rows)} IS_A relation rows.")

    print(f"All done. CSV files are in: {out_dir}")
