"""Extractive observation compressor: keep the sentences most relevant to the query within a token budget (≤180)."""
import re

_WORD = re.compile(r"\w+")


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


def compress_observation(text: str, query: str, max_tokens: int = 180) -> str:
    if approx_tokens(text) <= max_tokens:
        return text
    q = set(_WORD.findall(query.lower()))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    ranked = sorted(range(len(sents)), key=lambda i: -len(q & set(_WORD.findall(sents[i].lower()))))
    keep, used = set(), 0
    for i in ranked:
        t = approx_tokens(sents[i])
        if used + t <= max_tokens:
            keep.add(i); used += t
    return " ".join(sents[i] for i in sorted(keep))
