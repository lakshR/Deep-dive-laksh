"""Obsidian-vault link graph (design.md §1.2.4): atomic Markdown notes with YAML frontmatter and [[wikilinks]].
Traversal is tenant-scoped and capped at depth 2; only minimal node summaries are returned within a token budget."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.compress import approx_tokens

_WIKI = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_FM = re.compile(r"\A---\n(.*?)\n---\n", re.S)
SUMMARY_TOKENS = 240


@dataclass
class Note:
    slug: str
    meta: dict
    body: str
    links: set[str] = field(default_factory=set)


def _frontmatter(md: str) -> tuple[dict, str]:
    m = _FM.match(md)
    if not m:
        return {}, md
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, md[m.end():]


def summarize(body: str, max_tokens: int = SUMMARY_TOKENS) -> str:
    words, out = body.replace("#", "").split(), []
    for w in words:
        out.append(w)
        if approx_tokens(" ".join(out)) >= max_tokens:
            break
    return " ".join(out)


class VaultGraph:
    def __init__(self, tenant_id: str, vault: Path):
        self.tenant_id, self.notes = tenant_id, {}
        root = (vault / tenant_id).resolve()
        if root.is_dir() and root.parent == vault.resolve():
            for md in sorted(root.glob("*.md")):
                meta, body = _frontmatter(md.read_text())
                self.notes[md.stem] = Note(md.stem, meta, body.strip(), set(_WIKI.findall(body)))
        self.backlinks: dict[str, set[str]] = {}
        for n in self.notes.values():
            for t in n.links:
                self.backlinks.setdefault(t, set()).add(n.slug)

    def _slug(self, seed: str) -> str | None:
        parts = seed.split(":")
        s = parts[1] if len(parts) == 3 and parts[0] == self.tenant_id else seed   # chunk id -> doc stem
        return s if s in self.notes else None

    def expand(self, seeds: list[str], hops: int, token_budget: int, edge_types: list[str]) -> dict:
        use_back = "backlink" in edge_types
        frontier = [(s, None) for x in seeds if (s := self._slug(x))]
        seen = {s for s, _ in frontier}
        hop_nodes: list[tuple[str, str | None, int]] = []
        for hop in range(1, hops + 1):
            nxt = []
            for slug, _ in frontier:
                nbrs = (self.notes[slug].links if "wikilink" in edge_types else set()) | (self.backlinks.get(slug, set()) if use_back else set())
                for n in sorted(nbrs):
                    if n in self.notes and n not in seen:
                        seen.add(n); nxt.append((n, slug)); hop_nodes.append((n, slug, hop))
            frontier = nxt
        nodes, used, raw = [], 0, 0
        for slug, frm, hop in hop_nodes:
            note = self.notes[slug]
            raw += approx_tokens(note.body)
            summ = summarize(note.body)
            t = approx_tokens(summ)
            if used + t > token_budget:
                continue
            used += t
            nodes.append({"id": slug, "chunk_id": f"{self.tenant_id}:{slug}:0", "summary": summ, "edge_from": frm,
                          "hop": hop, "risk_tier": note.meta.get("risk_tier", "TIER_1_READ"),
                          "version": note.meta.get("version", "1")})
        return {"nodes": nodes, "tokens_used": used, "tokens_saved_vs_raw": max(0, raw - used)}
