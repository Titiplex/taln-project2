from typing import Iterable, List, Tuple

from spacy.language import Language
from spacy.tokens import Token


def _is_acronym(token: Token) -> bool:
    text = token.text
    if len(text) < 2 or len(text) > 10:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    return text.isupper()


def extract_acronyms(
        nlp: Language,
        sentences: Iterable[str],
) -> List[Tuple[int, str, str, str]]:
    """
    extracts a couple acronym / long form

    generated patterns :
      - Long Form (ACR)
      - ACR (Long Form)

    sens back a tuple list :
        (sent_id, acronym, long_form, sentence)
    """
    rows: List[Tuple[int, str, str, str]] = []

    for sent_id, doc in enumerate(nlp.pipe(sentences, batch_size=1000)):
        tokens = list(doc)
        n_tokens = len(tokens)
        sent_text = doc.text

        for i, tok in enumerate(tokens):
            if not _is_acronym(tok):
                continue
            acr = tok.text

            long_form = None

            # 1 pat. long form (ACR)
            if i + 1 < n_tokens and tokens[i + 1].text == ")":
                # looks for left parenthesis
                j = i - 1
                while j >= 0 and tokens[j].text != "(":
                    j -= 1
                if j >= 0:
                    long_tokens = [
                        t.text for t in tokens[j + 1: i] if t.is_alpha
                    ]
                    if long_tokens:
                        long_form = " ".join(long_tokens)

            # 2 pat. ACR (LONG FORM)
            if long_form is None and i + 1 < n_tokens and tokens[i + 1].text == "(":
                k = i + 2
                while k < n_tokens and tokens[k].text != ")":
                    k += 1
                if k < n_tokens and tokens[k].text == ")":
                    long_tokens = [
                        t.text for t in tokens[i + 2: k] if t.is_alpha
                    ]
                    if long_tokens:
                        long_form = " ".join(long_tokens)

            if long_form:
                rows.append((sent_id, acr, long_form, sent_text))

    return rows
