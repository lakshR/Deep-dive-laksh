# Load & evaluation tests

## What was actually run (this host: 4 CPUs, 15 GB, no GPU, free-tier LLM key)
| Tool | Run? | Result file |
|---|---|---|
| k6 (Docker `grafana/k6`) | **Yes — smoke only (3–10 VUs, ~30 s)** | `results/k6_*.json` |
| JMeter | **No** — plan written and checked as well-formed XML only | — |
| 50,000-VU k6 / JMeter run | **No** — infeasible here (see below) | — |

KPI-31…36 are defined **@ 50K VUs**; nothing in this folder certifies them. The workbench scorecard keeps them `NOT MEASURED`.

## Run
```bash
python load_tests/make_tokens.py            # per-tenant JWTs -> tokens.json / tokens.csv  (dev secret; gitignored)
docker run --rm --user $(id -u):$(id -g) --network host -v $PWD/load_tests:/work -w /work \
  -e VUS=10 -e DURATION=20s -e THINK=0.5 grafana/k6 run k6_load.js          # smoke
# cache-miss traffic: add -e UNIQUE=1      spec stages: -e PROFILE=spec50k   (needs a real cluster, see below)
jmeter -n -t load_tests/jmeter_plan.jmx -Jthreads=50 -Jramp=10 -Jduration=60 -l results/jmeter.jtl -e -o results/jmeter_dashboard
```

## Corrections to the blueprint's load model (design.md Appendix B #10, #11)
* The blueprint's k6 script uses tenant ids like `tenant_7`. Underscores are invalid under `TenantId`, so every request would return **401**. We use `tenant-7`.
* It sends `tenant_id`/`X-Tenant-ID` as if authoritative. They are only consistency assertions against the JWT, so the load test **mints per-tenant JWTs**.
* `sleep(1)` per VU makes the offered load ≈ VUs/s (≈50,000 RPS at 50K VUs) versus the KPI-35 target of ≥ 8,500 RPS. Think-time is a parameter (`THINK`); pick it deliberately and document it.
* Extra tenants (`make_tokens.py N`) have **empty vaults**: they exercise the gateway/isolation, not retrieval.

## Why 50K VUs cannot be certified from here
A single 4-CPU node serves ~5 RPS on the answer path (cross-encoder on CPU dominates; see below). The design assumes GPU ONNX-FP16 reranking, a Redis Cluster, Postgres replicas, HPA-scaled API pods and a fast LLM tier. The free-tier LLM key is capped at 8,000 tokens/min per model, so an LLM-path load test mostly measures the extractive fallback.

## Baseline measured on this host (k6, see `results/`)
Measured 2026-11-18, k6 via Docker, API + retrieval worker + load generator all on the same 4 CPUs, free-tier LLM key. 0 % HTTP failures in every run.

| Run | Rerank pool | VUs | p50 | p95 | p99 | RPS | note |
|---|---|---|---|---|---|---|---|
| mixed (most requests are cache misses) | 40 | 10 | 1090 ms | 1946 ms | 2165 ms | 5.5 | `k6_cached.json` |
| mixed | 10 | 10 | 986 ms | 1623 ms | 1868 ms | 5.8 | `k6_pool10_cachemix.json` |
| unique queries (all misses) | 40 | 3 | 373 ms | 980 ms | 1166 ms | – | LLM mostly rate-limited → extractive fallback |
| unique queries | 10 | 3 | 322 ms | 1081 ms | 1402 ms | 1.8 | |

Versus the targets (p50 ≤ 240, p95 ≤ 450, p99 ≤ 590 ms; ≥ 8,500 RPS @ 50K VUs): **not met**, by roughly 4× on latency and 3 orders of magnitude on throughput.

Where the time goes (span p50 under load, n≈200): dense embedding 230 ms (≈50 ms when run alone), rerank 246 ms, whole request 856 ms. Per-stage
times inflate 3–4× under concurrency: model inference is CPU-bound in a single retrieval worker process sharing 4 cores with the API and k6.
Cutting the rerank pool 40 → 10 made one query ~8× faster in isolation (`evals/rerank_pool.py`) but moved the loaded p50 only ~10 %.
Likely next levers (untested here): GPU / ONNX-FP16 inference (the design's plan), multiple retrieval workers, dynamic micro-batching, and a
dedicated host for the load generator. Also: `hook.pre` p95 was 24 ms (> the 15 ms KPI-01 gate) under load, p50 3 ms.
Re-run on target hardware before quoting any of these numbers.

## Evaluation sets (`evals/`)
`python evals/run_evals.py [--retrieval] [--promptguard]` → `evals/results/latest.json`. All corpora are **author-generated and seeded**
(reproducible). Each category also has a small **held-out, hand-written** set whose wording the detectors were *not* written against;
KPI-02/03 use the held-out numbers because they are the more realistic ones. Once a detector has been tuned against a held-out
item, that item is no longer held-out — the report says when this happened. The scorecard caps every eval-derived metric at
`INDICATIVE`: certifying a gate needs independently labelled data.
