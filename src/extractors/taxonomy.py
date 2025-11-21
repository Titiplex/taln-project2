from typing import Iterable, List, Tuple
import spacy


def extract_is_a(
        nlp: "spacy.language.Language",
        sentences: Iterable[str],
) -> List[Tuple[int, str, str, str, str]]:
    """
    extracts IS_A relations X is a Y / X is an Y

    returns tuple list :
        (sent_id, hyponym, relation, hypernym, sentence)
    """
    rows: List[Tuple[int, str, str, str, str]] = []

    for sent_id, doc in enumerate(nlp.pipe(sentences, batch_size=1000)):
        tokens = list(doc)
        n = len(tokens)
        sent_text = doc.text

        for i in range(0, n - 3):
            head = tokens[i]
            be = tokens[i + 1]
            art = tokens[i + 2]

            if be.lemma_ != "be":
                continue
            if art.lower_ not in {"a", "an"}:
                continue
            if head.pos_ not in {"PROPN", "NOUN"} and not head.ent_type_:
                continue

            # hypernyme -> fetching even the punct
            j = i + 3
            hyper_tokens = []
            while j < n and tokens[j].pos_ != "PUNCT":
                hyper_tokens.append(tokens[j])
                j += 1

            if not hyper_tokens:
                continue

            hyponym = head.text
            hypernym = " ".join(t.text for t in hyper_tokens).strip()
            if hypernym:
                rows.append((sent_id, hyponym, "IS_A", hypernym, sent_text))
                # one extraction per sentence
                break

    return rows
