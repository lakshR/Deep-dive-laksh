"""Heuristic prompt-injection scoring (KPI-03 target ≥ 99.5 %, to be measured on a large red-team corpus).
Normalizes (NFKC, zero-width removal, whitespace/homoglyph-light) then sums weighted pattern hits.
Used for user input (pre_execute) and at ingestion time for indirect injection in documents."""
import re
import unicodedata

_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
THRESHOLD = 0.6

_P = [  # (weight, regex)
    (0.7, r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the)\b.{0,30}\b(instruction|prompt|rule|guideline|direction|polic)"),
    (0.7, r"\b(reveal|show|print|output|repeat|leak|tell me)\b.{0,30}\b(system prompt|hidden prompt|initial prompt|(?:your|the)(?: initial| original| hidden| internal| full)? (?:instructions|rules|prompt)|developer message)"),
    (0.6, r"\byou are now\b|\bfrom now on,? you\b|\bact as (?:an? )?(?:unrestricted|jailbroken|dan|evil)\b|\bpretend (?:you are|to be)\b.{0,40}\b(no|without) (?:restriction|rule|filter|limit)"),
    (0.7, r"\b(developer|debug|god|jailbreak|dan)\s+mode\b|\bdo anything now\b"),
    (0.6, r"<\|?\s*(system|im_start|im_end|endoftext)\s*\|?>|\[\s*(system|inst)\s*\]|^\s*#{2,}\s*(system|instruction)s?\b"),
    (0.6, r"\b(disable|turn off|remove|skip)\b.{0,30}\b(safety|guardrail|filter|moderation|hitl|approval|redaction)"),
    (0.5, r"\bwithout (?:asking|requiring|waiting)(?: for)? (?:human )?(?:approval|review|confirmation)\b"),
    (0.6, r"\b(send|post|exfiltrate|upload|forward)\b.{0,40}\b(to|at)\b.{0,30}(https?://|\bwww\.)"),
    (0.4, r"\bnew instructions?\s*:|\bimportant\s*:\s*(?:ignore|disregard)|\bassistant\s*:\s*(?:sure|ok)"),
    (0.6, r"\b(show|give|list|dump|reveal|access|read|send|export|fetch)\b.{0,30}\b(other|another|all|every)\b.{0,15}\b(tenants?|customers?|users?|organi[sz]ations?|compan(?:y|ies))\W{0,3}s?\b.{0,12}\b(data|documents?|records?|secrets?|passwords?|files?|credentials?)"),
]
_RX = [(w, re.compile(p, re.I | re.S)) for w, p in _P]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_ZW)
    return re.sub(r"\s+", " ", t)


def score(text: str) -> tuple[float, list[int]]:
    t = normalize(text)
    hits = [i for i, (_, rx) in enumerate(_RX) if rx.search(t)]
    s = 1.0 - 1.0
    for i in hits:
        s = 1 - (1 - s) * (1 - _RX[i][0])   # noisy-OR combination
    return round(s, 3), hits


def is_injection(text: str, threshold: float = THRESHOLD) -> bool:
    return score(text)[0] >= threshold
