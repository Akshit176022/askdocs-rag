from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an the and or but if then of to in on at by for with from as is are was were be been
    being it its this that these those what which who whom how why when where do does did can
    could should would will shall may might i you he she we they them his her their our your
    not no yes so than too very also into about over under between during each any all some
    such only own same just there here more most other""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lower-case alphanumeric tokenizer. O(len(text))."""
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks
