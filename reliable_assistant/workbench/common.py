"""Shared helpers for the Streamlit workbench: API client (mints dev JWTs — dev only), palette, chart theme."""
import os
import sys
from pathlib import Path

import altair as alt
import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reliable_assistant.auth import mint_token  # noqa: E402

# dataviz skill reference palette (light): categorical slot 1, de-emphasis gray, status colors used only with icon+label
BLUE, GRAY, RED, GREEN, AMBER = "#2a78d6", "#b9b8b2", "#e34948", "#008300", "#c98500"
STATUS_ICON = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "INDICATIVE ✓": "◐ INDICATIVE ✓", "INDICATIVE ✗": "◐ INDICATIVE ✗",
               "NOT MEASURED": "⬜ NOT MEASURED"}


def themed(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(background="transparent").configure_axis(grid=True, gridColor="#8883", domainColor="#8886",
            labelColor="#8a8984", titleColor="#8a8984", labelFontSize=12, titleFontSize=12)
            .configure_view(strokeWidth=0).configure_legend(labelColor="#8a8984", titleColor="#8a8984"))


class Api:
    def __init__(self, base: str, tenant: str, user: str):
        self.base, self.tenant, self.user = base.rstrip("/"), tenant, user
        self.http = httpx.Client(base_url=self.base, timeout=120)

    def headers(self, role: str = "user", tenant: str | None = None, user: str | None = None) -> dict:
        return {"Authorization": f"Bearer {mint_token(tenant or self.tenant, user or self.user, role)}"}

    def get(self, path, role="developer", **params):
        return self.http.get(path, params=params or None, headers=self.headers(role))

    def post(self, path, role="developer", json=None, **kw):
        return self.http.post(path, json=json, headers=self.headers(role), **kw)

    def alive(self) -> dict | None:
        try:
            return self.http.get("/healthz", timeout=3).json()
        except Exception:  # noqa: BLE001
            return None


def api() -> Api:
    return st.session_state["api"]


def need_dev(r) -> dict | None:
    """Return JSON for a dev-endpoint response, or render a clear error."""
    if r.status_code == 404:
        st.error("Developer API is off. Start the server with `RA_DEV=1` (run_dev.sh does this).")
    elif r.status_code >= 400:
        st.error(f"HTTP {r.status_code}: {r.text[:300]}")
    else:
        return r.json()
    return None
