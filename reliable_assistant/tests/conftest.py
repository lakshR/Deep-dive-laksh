import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATE_PORT = 18081


@pytest.fixture(autouse=True)
def _isolated_wal(tmp_path, monkeypatch):
    import reliable_assistant.api as api
    monkeypatch.setattr(api, "WAL_PATH", tmp_path / "wal.jsonl")


def _up(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session")
def infra():
    if not (_up(5433) and _up(6390)):
        pytest.skip("docker compose services (postgres :5433, redis :6390) not running")
    proc = subprocess.Popen([sys.executable, str(ROOT / "mcp_servers" / "state_server.py")],
                            env={**os.environ, "RA_STATE_PORT": str(STATE_PORT)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if _up(STATE_PORT):
            break
        time.sleep(0.2)
    else:
        proc.kill(); pytest.skip("state server failed to start")
    import reliable_assistant.api as api
    api.STATE_URL = f"http://127.0.0.1:{STATE_PORT}/mcp"
    yield
    proc.terminate()
