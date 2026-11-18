"""Priority-ordered, fail-closed/fail-open plugin dispatch over pluggy (Appendix A)."""
import json
import logging
from pathlib import Path

import pluggy

from ..errors import AssistantError
from .specs import PROJECT, Specs, Verdict

log = logging.getLogger(__name__)
STAGE_HOOK = {"PRE_EXECUTION": "pre_execute", "TOOL_DISPATCH": "before_tool",
              "POST_EXECUTION": "post_execute", "ALL_SPANS": "on_span"}


class PluginManager:
    def __init__(self, config: dict):
        self.pm = pluggy.PluginManager(PROJECT)
        self.pm.add_hookspecs(Specs)
        self.config = {p["name"]: p for p in config.get("plugins", [])}
        self.enabled: dict[str, bool] = {n: True for n in self.config}

    @classmethod
    def from_file(cls, path: str | Path) -> "PluginManager":
        return cls(json.loads(Path(path).read_text()))

    def register(self, name: str, plugin: object) -> None:
        if name not in self.config:
            raise ValueError(f"plugin {name!r} not declared in mcp_config.json")
        self.pm.register(plugin, name=name)

    def set_enabled(self, name: str, on: bool) -> None:  # hot toggle (Streamlit Tab 3)
        self.enabled[name] = on

    def _ordered(self, stage: str) -> list[str]:
        names = [n for n, c in self.config.items()
                 if c["stage"] == stage and self.enabled[n] and self.pm.get_plugin(n) is not None]
        return sorted(names, key=lambda n: self.config[n].get("priority", 100))

    def _call(self, name: str, hook: str, **kw):
        """Call only plugin `name`'s impl of `hook`."""
        others = [p for n, p in self.pm.list_name_plugin() if n != name]
        caller = self.pm.subset_hook_caller(hook, remove_plugins=others)
        return caller(**kw)

    def run_verdicts(self, stage: str, **kw) -> dict:
        """Run verdict hooks in ascending priority; first deny short-circuits.
        Returns accumulated ctx_updates. Raises AssistantError on deny / fail-closed fault."""
        hook, updates = STAGE_HOOK[stage], {}
        for name in self._ordered(stage):
            try:
                results = [r for r in self._call(name, hook, **kw) if r is not None]
            except Exception as exc:  # noqa: BLE001
                if self.config[name].get("fail_closed", False):
                    raise AssistantError(-32603, f"plugin {name} failed closed: {exc}") from exc
                log.warning("plugin %s failed open: %s", name, exc)
                continue
            for v in results:
                if not v.allow:
                    raise AssistantError(v.code or -32012, f"{name}: {v.reason}", v.data)
                updates.update(v.ctx_updates)
            if "ctx" in kw:
                kw["ctx"] = {**kw["ctx"], **updates}
        return updates

    def run_observers(self, stage: str, **kw) -> None:
        """post_execute / on_span: fail-open, never raise."""
        hook = STAGE_HOOK[stage]
        for name in self._ordered(stage):
            try:
                self._call(name, hook, **kw)
            except Exception as exc:  # noqa: BLE001
                log.warning("observer %s failed: %s", name, exc)
