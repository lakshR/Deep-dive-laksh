// k6 load test — design.md Appendix C. Defaults to a SMOKE profile; PROFILE=spec50k reproduces the 50K-VU stages.
//   docker run --rm --network host -v $PWD/load_tests:/work -w /work grafana/k6 run k6_load.js
// Env: PROFILE=smoke|spec50k  VUS  DURATION  THINK (seconds)  UNIQUE=1 (cache-miss queries)  BASE_URL
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

const tokens = new SharedArray('tokens', () => JSON.parse(open('./tokens.json')));
const BASE = __ENV.BASE_URL || 'http://localhost:8000';
const THINK = parseFloat(__ENV.THINK || '0.5');   // Appendix B #10: think-time sets the offered load; document it.
const UNIQUE = __ENV.UNIQUE === '1';
const QUERIES = ['How fast must recalled lots be quarantined?', 'What are the SKU labeling rules?',
                 'Who approves refunds above 100 dollars?', 'What is the lot recall procedure?'];

const spec = [{ duration: '2m', target: 10000 }, { duration: '3m', target: 50000 },
              { duration: '5m', target: 50000 }, { duration: '2m', target: 0 }];
const smoke = [{ duration: '5s', target: +(__ENV.VUS || 10) }, { duration: __ENV.DURATION || '20s', target: +(__ENV.VUS || 10) },
               { duration: '3s', target: 0 }];

export const options = {
  stages: (__ENV.PROFILE === 'spec50k') ? spec : smoke,
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    http_req_duration: ['p(50)<250', 'p(95)<450', 'p(99)<600'],   // KPI-31..33 (gate: p99 <= 590)
    http_req_failed: ['rate<0.0005'],                              // KPI-34
    checks: ['rate>0.999'],
  },
};

export default function () {
  const t = tokens[(__VU * 7 + __ITER) % tokens.length];
  let q = QUERIES[__ITER % QUERIES.length];
  if (UNIQUE) q += ` (${__VU}-${__ITER}-${Date.now()})`;
  const res = http.post(`${BASE}/api/v1/agent/query`, JSON.stringify({ message: q, tenant_id: t.tenant }), {
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${t.token}`, 'X-Tenant-ID': t.tenant },
  });
  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency within 600ms SLA': (r) => r.timings.duration <= 600,
    'no cross-tenant leak (tenant echo)': (r) => r.status === 200 && r.json('tenant_id') === t.tenant,
  });
  sleep(THINK);
}

export function handleSummary(data) {
  return { '/work/results/k6_summary.json': JSON.stringify(data, null, 2) };
}
