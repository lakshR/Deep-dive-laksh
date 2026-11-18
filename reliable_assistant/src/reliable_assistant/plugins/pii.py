"""Regex PII redaction (KPI-02 target ≥ 99.8 %, to be measured on a large corpus — see design.md Appendix B #9).
Covers email (incl. obfuscated/spaced), phone, SSN, payment cards (Luhn), IBAN, IPv4/IPv6. Does NOT detect free-text
names, addresses, passport numbers or dates of birth; those need NER (Presidio) — a documented follow-up."""
import re
import unicodedata

_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"))

_SYM_AT = r"(?:@|\s*[\[\(\{<]\s*at\s*[\]\)\}>]\s*)"
_SYM_DOT = r"(?:\.|\s*[\[\(\{<]\s*dot\s*[\]\)\}>]\s*)"
_WORD_DOT = r"\s+dot\s+"
EMAIL = re.compile(rf"[A-Za-z0-9._%+-]+{_SYM_AT}[A-Za-z0-9-]+(?:{_SYM_DOT}[A-Za-z0-9-]+)*{_SYM_DOT}[A-Za-z]{{2,}}\b", re.I)
# fully spelled-out form requires BOTH 'at' and 'dot' as words, so 'mail me at john.doe@x.com' is not misparsed
EMAIL_SPACED = re.compile(r"[\w.+%-]+\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+")          # 'john.doe @ acme . com'
EMAIL_WORDS = re.compile(rf"\b[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+|\s+dot\s+[A-Za-z0-9_%+-]+)*\s+at\s+[A-Za-z0-9-]+(?:{_WORD_DOT}[A-Za-z0-9-]+)*{_WORD_DOT}[A-Za-z]{{2,}}\b", re.I)
SSN = re.compile(r"\b\d{3}[-\s.]\d{2}[-\s.]\d{4}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w-])")
DIGITS_SPACED = re.compile(r"(?<![\w-])\+?(?:\d{1,4}[\s.-]){2,14}\d{1,4}(?![\w-])")                        # '4 1 5 5 5 5 0 1 3 2'
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{1,4}(?![\w:])")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")


def _luhn(num: str) -> bool:
    d = [int(c) for c in num if c.isdigit()][::-1]
    return len(d) >= 13 and sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(d)) % 10 == 0


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    t = unicodedata.normalize("NFKC", text).translate(_ZW)

    def sub(rx, label, t, check=None):
        def f(m):
            if check and not check(m.group(0)):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"
        return rx.sub(f, t)

    t = sub(EMAIL, "EMAIL", t)
    t = sub(EMAIL_WORDS, "EMAIL", t)
    t = sub(EMAIL_SPACED, "EMAIL", t)
    t = sub(IBAN, "IBAN", t)
    t = sub(IPV6, "IP", t, lambda x: "::" in x or re.search(r"[a-fA-F]", x))   # not clock times like 10:30:15
    t = sub(SSN, "SSN", t)
    t = sub(CARD, "CARD", t, _luhn)
    t = sub(IPV4, "IP", t)
    t = sub(PHONE, "PHONE", t, lambda s: 10 <= sum(c.isdigit() for c in s) <= 15)
    t = sub(DIGITS_SPACED, "PHONE", t, lambda s: 10 <= sum(c.isdigit() for c in s) <= 15)
    return t, counts
