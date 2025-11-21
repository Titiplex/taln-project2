from typing import Iterable, List, Tuple
import spacy


def extract_named_entities(
        nlp: "spacy.language.Language",
        sentences: Iterable[str],
) -> List[Tuple[int, str, str, str, str]]:
    """
    Extracts named entities.

    returns a list of tuples :
        (sent_id, entity_text, entity_label, normalized_form, sentence)
    """
    rows: List[Tuple[int, str, str, str, str]] = []
    for i, doc in enumerate(nlp.pipe(sentences, batch_size=1000)):
        sent_text = doc.text
        for ent in doc.ents:
            text = ent.text.strip()
            if not text:
                continue
            # simple norm : lower + squash spaces
            norm = " ".join(text.split()).lower()
            rows.append((i, text, ent.label_, norm, sent_text))
    return rows
