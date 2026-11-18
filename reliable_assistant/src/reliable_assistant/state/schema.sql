-- Reliable Assistant state schema (design.md §1.2.5, §1.3). Applied by the admin role; the app connects as ra_app,
-- which is NOT a superuser and cannot bypass RLS. Every row carries tenant_id; every query runs under
-- SET LOCAL app.tenant_id.
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ra_app') THEN
    CREATE ROLE ra_app LOGIN PASSWORD 'ra_app_dev_only' NOSUPERUSER NOBYPASSRLS;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS session_facts (
  id BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, user_hash TEXT NOT NULL,
  fact TEXT NOT NULL CHECK (char_length(fact) <= 300), source_chunk_id TEXT NOT NULL,
  verified BOOLEAN NOT NULL DEFAULT TRUE CHECK (verified), created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS session_facts_lookup ON session_facts (tenant_id, session_id);

CREATE TABLE IF NOT EXISTS hitl_tickets (
  ticket_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_hash TEXT NOT NULL, workflow TEXT NOT NULL,
  payload JSONB NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING_APPROVAL'
    CHECK (status IN ('PENDING_APPROVAL','APPROVED','REJECTED')),
  requested_at TIMESTAMPTZ NOT NULL DEFAULT now(), decided_by TEXT, decided_at TIMESTAMPTZ);
CREATE INDEX IF NOT EXISTS hitl_pending ON hitl_tickets (tenant_id, status);

CREATE TABLE IF NOT EXISTS audit_log (
  tenant_id TEXT NOT NULL, seq BIGINT NOT NULL, prev_hash CHAR(64) NOT NULL, record_hash CHAR(64) NOT NULL,
  payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (tenant_id, seq));

DO $$ DECLARE t TEXT; BEGIN
  FOREACH t IN ARRAY ARRAY['session_facts','hitl_tickets','audit_log'] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I', t);
    EXECUTE format($p$CREATE POLICY tenant_isolation ON %I USING (tenant_id = current_setting('app.tenant_id', true))
                      WITH CHECK (tenant_id = current_setting('app.tenant_id', true))$p$, t);
  END LOOP; END $$;

GRANT USAGE ON SCHEMA public TO ra_app;
GRANT SELECT, INSERT ON session_facts, audit_log TO ra_app;     -- audit is append-only: no UPDATE/DELETE
GRANT SELECT, INSERT, UPDATE ON hitl_tickets TO ra_app;
GRANT USAGE ON SEQUENCE session_facts_id_seq TO ra_app;
