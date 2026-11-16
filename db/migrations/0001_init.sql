-- 0001_init.sql — foundation schema (brief §6, workstream A).
-- Forward-only. Never edit an applied migration; add the next number.
--
-- Conventions:
--   * text + CHECK instead of Postgres enums (cheap to extend, explicit contract).
--   * timestamptz everywhere; money is numeric with an explicit currency column.
--   * JSONB only where the brief says *_json; nothing is parsed in the database.
--   * claims are append-only. Corrections insert and point `superseded_by` at
--     the row they replace.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------- registry
-- §6 does not list registry tables, but `fmd.cli seed` must load countries,
-- sources and exams (deliverable §7). These are config projections, never
-- user data; config/*.yaml remains the source of truth.

CREATE TABLE IF NOT EXISTS countries (
    code             text PRIMARY KEY,
    name             text NOT NULL,
    enabled          boolean NOT NULL DEFAULT false,
    currency         char(3) NOT NULL,
    tier             text NOT NULL,
    languages_json   jsonb NOT NULL DEFAULT '[]'::jsonb,
    capabilities_json jsonb,
    updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sources (
    id                    text PRIMARY KEY,
    name                  text NOT NULL,
    base_url              text,
    kind                  text NOT NULL,
    format                text NOT NULL,
    cadence_days          integer NOT NULL CHECK (cadence_days > 0),
    trust                 text NOT NULL CHECK (trust IN ('high', 'medium', 'low')),
    adapter               text NOT NULL,
    enabled               boolean NOT NULL DEFAULT false,
    notes                 text,
    requires_manual_review boolean NOT NULL DEFAULT false,
    seeds_json            jsonb NOT NULL DEFAULT '[]'::jsonb,
    config_json           jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS source_exclusions (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    reason      text NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- --------------------------------------------------------------------------- entities
CREATE TABLE IF NOT EXISTS entities (
    id              bigserial PRIMARY KEY,
    kind            text NOT NULL
                    CHECK (kind IN ('institution', 'programme', 'hospital', 'exam',
                                    'lender', 'city')),
    canonical_name  text NOT NULL,
    country         char(2),
    state           text,
    city            text,
    status          text NOT NULL DEFAULT 'active',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS entities_kind_name_idx ON entities (kind, canonical_name);
CREATE INDEX IF NOT EXISTS entities_country_state_idx ON entities (country, state);

CREATE TABLE IF NOT EXISTS entity_aliases (
    id          bigserial PRIMARY KEY,
    entity_id   bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    alias       text NOT NULL,
    alias_type  text NOT NULL
                CHECK (alias_type IN ('acronym', 'former_name', 'transliteration',
                                      'misspelling', 'local_name')),
    source_id   text REFERENCES sources (id),
    confidence  double precision CHECK (confidence IS NULL OR
                                        (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX IF NOT EXISTS entity_aliases_entity_idx ON entity_aliases (entity_id);
CREATE UNIQUE INDEX IF NOT EXISTS entity_aliases_unique_idx
    ON entity_aliases (entity_id, alias, alias_type);

-- --------------------------------------------------------------------------- evidence
CREATE TABLE IF NOT EXISTS documents (
    id                  bigserial PRIMARY KEY,
    url                 text NOT NULL UNIQUE,
    domain              text NOT NULL,
    kind                text,
    first_seen          timestamptz,
    last_seen           timestamptz,
    etag                text,
    last_modified       timestamptz,
    robots_decision     text,
    robots_checked_at   timestamptz,
    http_status         integer
);

CREATE INDEX IF NOT EXISTS documents_domain_idx ON documents (domain);
CREATE INDEX IF NOT EXISTS documents_last_seen_idx ON documents (last_seen DESC);

CREATE TABLE IF NOT EXISTS snapshots (
    id              bigserial PRIMARY KEY,
    document_id     bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    sha256          text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    bytes_path      text NOT NULL,
    fetched_at      timestamptz NOT NULL DEFAULT now(),
    fetcher_tier    text,
    content_type    text,
    headers_json    jsonb,
    UNIQUE (document_id, sha256)
);

CREATE INDEX IF NOT EXISTS snapshots_document_idx ON snapshots (document_id);
CREATE INDEX IF NOT EXISTS snapshots_sha256_idx ON snapshots (sha256);

CREATE TABLE IF NOT EXISTS extractions (
    id              bigserial PRIMARY KEY,
    snapshot_id     bigint NOT NULL REFERENCES snapshots (id) ON DELETE CASCADE,
    tool            text NOT NULL,
    tool_version    text,
    model           text,
    prompt_version  text,
    page_range      text,
    text            text NOT NULL,
    structure_json  jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS extractions_snapshot_idx ON extractions (snapshot_id);

CREATE TABLE IF NOT EXISTS claims (
    id              bigserial PRIMARY KEY,
    extraction_id   bigint NOT NULL REFERENCES extractions (id) ON DELETE RESTRICT,
    entity_id       bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    predicate       text NOT NULL,
    value_json      jsonb NOT NULL,
    unit            text,
    qualifier_json  jsonb,
    quote           text NOT NULL,
    char_start      integer NOT NULL CHECK (char_start >= 0),
    char_end        integer NOT NULL CHECK (char_end >= char_start),
    confidence      double precision NOT NULL DEFAULT 1.0
                    CHECK (confidence >= 0 AND confidence <= 1),
    observed_at     timestamptz,
    method          text NOT NULL
                    CHECK (method IN ('extracted', 'derived', 'modelled', 'user_provided')),
    superseded_by   bigint REFERENCES claims (id) ON DELETE SET NULL,
    CHECK (superseded_by IS NULL OR superseded_by <> id)
);

CREATE INDEX IF NOT EXISTS claims_entity_predicate_idx ON claims (entity_id, predicate);
CREATE INDEX IF NOT EXISTS claims_extraction_idx ON claims (extraction_id);
CREATE INDEX IF NOT EXISTS claims_observed_idx ON claims (observed_at DESC);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              bigserial PRIMARY KEY,
    purpose         text NOT NULL,
    prompt_sha256   text NOT NULL,
    prompt_path     text,
    model           text NOT NULL,
    provider        text NOT NULL,
    params_json     jsonb,
    raw_response    text,
    tokens_in       integer CHECK (tokens_in IS NULL OR tokens_in >= 0),
    tokens_out      integer CHECK (tokens_out IS NULL OR tokens_out >= 0),
    latency_ms      integer CHECK (latency_ms IS NULL OR latency_ms >= 0),
    cost_usd        numeric(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    cache_hit       boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS llm_calls_purpose_idx ON llm_calls (purpose, created_at DESC);

-- --------------------------------------------------------------------------- programmes
CREATE TABLE IF NOT EXISTS programmes (
    id                      bigserial PRIMARY KEY,
    institution_id          bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    name                    text NOT NULL,
    credential_level        text NOT NULL
                            CHECK (credential_level IN ('degree', 'diploma',
                                                        'lateral_entry', 'certificate')),
    duration_months         integer CHECK (duration_months IS NULL OR duration_months > 0),
    internship_months       numeric(4, 1)
                            CHECK (internship_months IS NULL OR internship_months >= 0),
    affiliated_university_id bigint REFERENCES entities (id) ON DELETE SET NULL,
    recognition_status      text NOT NULL DEFAULT 'unknown'
                            CHECK (recognition_status IN ('recognised', 'conditional',
                                                          'unrecognised', 'unknown')),
    seats                   integer CHECK (seats IS NULL OR seats >= 0),
    medium                  text,
    created_at              timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS programmes_institution_idx ON programmes (institution_id);
CREATE INDEX IF NOT EXISTS programmes_credential_idx ON programmes (credential_level);

CREATE TABLE IF NOT EXISTS scores (
    id              bigserial PRIMARY KEY,
    entity_id       bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    scorer_version  text NOT NULL,
    weights_version text NOT NULL,
    claims_hash     text NOT NULL,
    computed_at     timestamptz NOT NULL DEFAULT now(),
    sub_scores_json jsonb NOT NULL,
    composite       double precision CHECK (composite IS NULL OR
                                            (composite >= 0 AND composite <= 100)),
    composite_lo    double precision CHECK (composite_lo IS NULL OR
                                            (composite_lo >= 0 AND composite_lo <= 100)),
    composite_hi    double precision CHECK (composite_hi IS NULL OR
                                            (composite_hi >= 0 AND composite_hi <= 100)),
    evidence_tier   text NOT NULL
                    CHECK (evidence_tier IN ('well_evidenced', 'partial', 'thin',
                                             'insufficient')),
    CHECK (composite_lo IS NULL OR composite_hi IS NULL OR composite_lo <= composite_hi)
);

CREATE INDEX IF NOT EXISTS scores_entity_computed_idx ON scores (entity_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS scores_claims_hash_idx ON scores (claims_hash);

CREATE TABLE IF NOT EXISTS insight_sets (
    id              bigserial PRIMARY KEY,
    entity_id       bigint NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    scores_id       bigint NOT NULL REFERENCES scores (id) ON DELETE CASCADE,
    insight_ids_json jsonb NOT NULL,
    coverage_json   jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS insight_sets_scores_idx ON insight_sets (scores_id);

-- --------------------------------------------------------------------------- crawling
CREATE TABLE IF NOT EXISTS crawl_frontier (
    url             text PRIMARY KEY,
    domain          text NOT NULL,
    priority        integer NOT NULL DEFAULT 0,
    next_allowed_at timestamptz NOT NULL DEFAULT now(),
    attempts        integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    state           text NOT NULL DEFAULT 'queued'
                    CHECK (state IN ('queued', 'in_progress', 'done', 'failed',
                                     'quarantined')),
    tier_used       text,
    ban_score       double precision CHECK (ban_score IS NULL OR
                                            (ban_score >= 0 AND ban_score <= 1)),
    yield_score     double precision CHECK (yield_score IS NULL OR
                                            (yield_score >= 0 AND yield_score <= 1)),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS crawl_frontier_ready_idx
    ON crawl_frontier (next_allowed_at, state);
CREATE INDEX IF NOT EXISTS crawl_frontier_domain_idx ON crawl_frontier (domain);

CREATE TABLE IF NOT EXISTS domain_politeness (
    domain              text PRIMARY KEY,
    rps                 double precision CHECK (rps IS NULL OR rps > 0),
    crawl_delay         double precision CHECK (crawl_delay IS NULL OR crawl_delay >= 0),
    robots_json         jsonb,
    robots_fetched_at   timestamptz,
    quarantined_until   timestamptz
);

-- --------------------------------------------------------------------------- exams
CREATE TABLE IF NOT EXISTS exams (
    id                  text PRIMARY KEY,
    name                text NOT NULL,
    country             char(2) NOT NULL REFERENCES countries (code),
    scope               text NOT NULL
                        CHECK (scope IN ('national', 'state', 'institution')),
    provider            text NOT NULL,
    registration_opens  timestamptz,
    registration_closes timestamptz,
    earliest_sitting    date,
    latest_sitting      date,
    requires_travel     boolean NOT NULL DEFAULT true,
    remote_options      boolean NOT NULL DEFAULT false,
    fee_amount          numeric(12, 2) CHECK (fee_amount IS NULL OR fee_amount >= 0),
    currency            char(3),
    source_claim_id     bigint REFERENCES claims (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS exams_country_scope_idx ON exams (country, scope);

CREATE TABLE IF NOT EXISTS exam_windows (
    id                  bigserial PRIMARY KEY,
    exam_id             text NOT NULL REFERENCES exams (id) ON DELETE CASCADE,
    cycle_year          integer NOT NULL,
    registration_opens  timestamptz,
    registration_closes timestamptz,
    sitting_dates_json  jsonb,
    source_claim_id     bigint REFERENCES claims (id) ON DELETE SET NULL,
    observed_at         timestamptz NOT NULL DEFAULT now(),
    UNIQUE (exam_id, cycle_year)
);

-- --------------------------------------------------------------------------- cost + financing
CREATE TABLE IF NOT EXISTS cost_items (
    id              bigserial PRIMARY KEY,
    programme_id    bigint NOT NULL REFERENCES programmes (id) ON DELETE CASCADE,
    category        text NOT NULL
                    CHECK (category IN ('tuition', 'hostel', 'mess', 'rent', 'utilities',
                                        'food', 'transport', 'equipment', 'exam_fee',
                                        'visa', 'insurance', 'misc')),
    label           text NOT NULL,
    amount          numeric(14, 2) NOT NULL CHECK (amount >= 0),
    currency        char(3) NOT NULL,
    period          text NOT NULL
                    CHECK (period IN ('one_time', 'per_year', 'per_month')),
    basis           text,
    source_claim_id bigint REFERENCES claims (id) ON DELETE SET NULL,
    confidence      double precision CHECK (confidence IS NULL OR
                                            (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX IF NOT EXISTS cost_items_programme_idx ON cost_items (programme_id);

CREATE TABLE IF NOT EXISTS financing_options (
    id                  bigserial PRIMARY KEY,
    scope               text NOT NULL CHECK (scope IN ('domestic', 'overseas')),
    lender              text NOT NULL,
    scheme              text NOT NULL,
    rate_min            numeric(6, 3) CHECK (rate_min IS NULL OR rate_min >= 0),
    rate_max            numeric(6, 3) CHECK (rate_max IS NULL OR rate_max >= 0),
    collateral_rules    text,
    max_amount          numeric(14, 2) CHECK (max_amount IS NULL OR max_amount >= 0),
    moratorium_months   integer CHECK (moratorium_months IS NULL OR moratorium_months >= 0),
    tenure_months       integer CHECK (tenure_months IS NULL OR tenure_months > 0),
    subsidy_rules       text,
    source_claim_id     bigint REFERENCES claims (id) ON DELETE SET NULL,
    observed_at         timestamptz NOT NULL DEFAULT now(),
    CHECK (rate_min IS NULL OR rate_max IS NULL OR rate_min <= rate_max)
);

CREATE INDEX IF NOT EXISTS financing_options_scope_idx ON financing_options (scope, scheme);
