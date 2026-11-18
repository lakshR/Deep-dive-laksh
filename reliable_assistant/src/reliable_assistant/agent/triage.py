"""Deterministic rule-based triage (§1.2.6). PLACEHOLDER for the distilled classifier: KPI-04 (≥97 %) must be
measured on a labeled set before this is accepted. Pure regex, so well within the 25 ms budget (KPI-05)."""
import re

from ..types import TriageOut

_QUESTION = re.compile(r"^\s*(what|how|when|why|which|who|where|is|are|does|do|did|can you tell|tell me|explain|describe|show)\b", re.I)
_ACTION = re.compile(r"\b(approve|submit|transfer|pay|refund|adjust|cancel|release|authorize|prescribe|delete|issue|reorder)\b", re.I)
_REQUEST = re.compile(r"^\s*(please\b|i (want|need|would like) to\b|can you\b|could you\b|go ahead and\b|let'?s\b)|^\s*(approve|submit|transfer|pay|refund|adjust|cancel|release|authorize|prescribe|delete|issue|reorder)\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b[\s\w!.,]{0,30}$", re.I)
_OOS = re.compile(r"\b(weather|joke|poem|recipe|horoscope|lottery|movie)\b", re.I)
_DOMAINS = {
    "clinical": re.compile(r"\b(dose|dosage|patient|clinical|prescri\w+|medication|diagnos\w+)\b", re.I),
    "financial": re.compile(r"\b(refund|payment|pay|invoice|transfer|budget|account balance|price)\b", re.I),
    "legal": re.compile(r"\b(contract|legal|liability|compliance|lawsuit|clause)\b", re.I),
    "inventory": re.compile(r"\b(sku|lot|batch|stock|inventory|warehouse|quarantine|recall)\b", re.I),
}


_NOUN_AFTER = re.compile(r"^\s*(?:approve|submit|transfer|pay|refund|adjust|cancel|release|authorize|prescribe|delete|issue|reorder)\s+(?:policy|policies|procedure|process|rules?|summary|guidelines?|workflow|history|status|limits?|options?)\b", re.I)


def triage(message: str) -> TriageOut:
    domain = next((d for d, rx in _DOMAINS.items() if rx.search(message)), "general")
    words = len(message.split())
    complexity = "low" if words <= 12 else "medium" if words <= 40 else "high"
    if _GREETING.match(message):
        return TriageOut(intent="smalltalk", domain="general", risk_tier="TIER_1_READ", complexity="low", route="DIRECT_ANSWER")
    if _OOS.search(message) and domain == "general":
        return TriageOut(intent="out_of_scope", domain="other", risk_tier="TIER_1_READ", complexity="low", route="REJECT_OOS")
    if _REQUEST.search(message) and _ACTION.search(message) and not _QUESTION.match(message) and not _NOUN_AFTER.match(message):
        verb = _ACTION.search(message).group(1).lower()
        return TriageOut(intent=f"{verb}_request", domain=domain, risk_tier="TIER_3_MUTATE",
                         complexity=complexity, route="TRANSACTION_EXEC")
    return TriageOut(intent="knowledge_lookup", domain=domain, risk_tier="TIER_1_READ",
                     complexity=complexity, route="HYBRID_RAG")
