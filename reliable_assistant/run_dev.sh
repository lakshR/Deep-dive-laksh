#!/usr/bin/env bash
# Dev launcher: Redis + Postgres (docker), schema, mcp-state-store (:8081), API (:8000). Logs in ./var/.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p var
docker compose up -d
until docker exec ra-postgres pg_isready -U ra -q; do sleep 1; done
PYTHONPATH=src python3 -c "import asyncio;from reliable_assistant.state.store import apply_schema;asyncio.run(apply_schema('postgresql://ra:ra_dev_only@localhost:5433/ra'))"
setsid nohup python3 mcp_servers/state_server.py > var/state.log 2>&1 &
for _ in $(seq 50); do (echo > /dev/tcp/127.0.0.1/8081) 2>/dev/null && break; sleep 0.3; done
RA_DEV=1 RA_RERANK_POOL=${RA_RERANK_POOL:-10} PYTHONPATH=src setsid nohup python3 -m uvicorn reliable_assistant.api:app --port 8000 > var/api.log 2>&1 &
setsid nohup python3 -m streamlit run app.py --server.port 8501 --server.headless true > var/workbench.log 2>&1 &
setsid nohup python3 -m streamlit run chat_app.py --server.port 8502 --server.headless true > var/chat.log 2>&1 &
echo "state-store :8081, API :8000 (RA_DEV=1), workbench :8501, chat app http://localhost:8502 (logs in var/)"
