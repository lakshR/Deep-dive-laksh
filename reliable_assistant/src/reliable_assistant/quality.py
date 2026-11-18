"""Post-execution answer audit — the `ragas_auditor` plugin's implementation.
This is a cheap LEXICAL PROXY, not RAGAS: real RAGAS needs several LLM calls per answer (infeasible on the current
rate-limited key) and is better run offline over audit logs. Scores:
  groundedness  = share of the answer's content tokens that appear in the cited evidence
  entity_recall = share of identifiers (LOT-/SKU-) in the question that the answer/evidence covers"""
import re

from .plugins.injection import normalize
from .retrieval.index import tokenize   # same Porter stemming as the BM25 stage

_STOP = set("the a an and or of to in on for with is are be by from at as that this it its must may all any after before "
            "when which who what how not no if then than also can will".split())
_ID = re.compile(r"\b(?:LOT-\d{5}|SKU-\d{3,6})\b")
_TOK = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _content(text: str) -> set[str]:
    raw = [t for t in _TOK.findall(normalize(text).lower()) if t not in _STOP and len(t) > 2]
    return set(tokenize(" ".join(raw)))


def audit_answer(question: str, answer: str, evidence_texts: list[str]) -> dict:
    ans, ev = _content(answer), set().union(*[_content(t) for t in evidence_texts]) if evidence_texts else set()
    grounded = (len(ans & ev) / len(ans)) if ans else 0.0
    ids = set(_ID.findall(question))
    covered = {i for i in ids if i in answer or any(i in t for t in evidence_texts)}
    return {"groundedness": round(grounded, 3), "entity_recall": round(len(covered) / len(ids), 3) if ids else None}


class RagasAuditorImpl:
    def __init__(self, cfg: dict | None = None):
        self.min_grounded = (cfg or {}).get("min_lexical_groundedness", 0.8)

    def run(self, ctx: dict, answer: dict) -> None:
        scores = audit_answer(ctx.get("message", ""), answer.get("answer", ""), ctx.get("evidence_texts", []))
        scores["needs_review"] = bool(answer.get("citations")) and scores["groundedness"] < self.min_grounded
        ctx["post_audit"] = scores           # ctx is shared by reference: the API reads this to gate cache writes
