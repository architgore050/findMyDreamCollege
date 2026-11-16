# Interface contract (derived from code, not the brief)

Other workstreams implement against these names/types. Schema changes need a new
`db/migrations/NNNN_*.sql`; never edit `0001_init.sql`. Each section names the
workstream that owns it.

## SourceAdapter / DocumentRef — WS-A/WS-B (`fmd/adapters/__init__.py`)
```python
class DocumentRef(Protocol):  # fetch layer only
    url: str
    source_id: str | None
    etag: str | None            # optional; previous response validator
    last_modified: str | None   # optional; HTTP-date string (If-Modified-Since)
    priority: int               # optional frontier hint, default 0


class SourceAdapter(Protocol):  # fmd/adapters/<family>.py
    name: str

    def fetch(self) -> Iterable[DocumentRef]: ...
```
`fmd/discover/base.py:DocumentRef` is the concrete dataclass; every new field is
optional with a default, so existing constructors keep working. `etag` /
`last_modified` let WS-C issue a conditional GET without re-reading
`DocumentsRepo`; `FetchRequest.revalidate(url, etag=..., last_modified=...)`
accepts the stored `datetime` and renders the header value (`http_date`).
`config/sources.yaml:adapter` must be in `KNOWN_ADAPTERS`; unknown names fail registry load.
Adapters that abstain (`verify_first`, `abstain`) raise `SourceUnavailable` with a
reason — never an empty success.

## Evidence store — WS-A (`fmd/store/snapshot_store.py`)
`SnapshotStore(root)` → `put(bytes)->sha256`, `get(sha256)->bytes`, `verify(sha256)->bool`,
`exists(sha256)->bool`, `path_for(sha256)->Path`; atomic, content-addressed
(`objects/<2hex>/<sha256>`), refuses to overwrite corrupt evidence. DB row via
`SnapshotsRepo.create`.

## Extract pipeline — WS-D (`fmd/extract/pipeline.py`)
```python
ExtractPipeline(
    router, *, snapshots, extractions, claims, store,
    tool="fmd.extract", tool_version="1.0.0",
    policy: TierPolicy | None = None, bands: Bands | None = None,
    observed_at: datetime | None = None, default_confidence: float = 0.5,
    adjudicate: bool = False, adjudicator: VLMAdjudicator | None = None,
)
.run(ExtractTarget(snapshot_id, entity_id, family)) -> ExtractResult
.run_batch(list[ExtractTarget]) -> list[ExtractResult]   # sorted, resumable
```
`run_batch` orders targets by `(snapshot_id, family)` and `run` skips a snapshot
already extracted at the same `tool_version` + `prompt_version`
(`skipped=True`). `ExtractResult` carries `extraction_id`, `doc`, stored
`claims`, `rejections`, `rewrites`, `adjudications`.
`build_router(models=None, *, mock=False, fixtures_dir=None)` is the offline/CI
factory: `mock=True` re-points every profile at the replay provider
(`mock_models`); it raises `ExtractError` when no fixtures directory is configured.
`run_extract(targets, *, database, store, registry=None, router=None, mock=False,
fixtures_dir=None, **pipeline_kwargs)` is the `make extract` seam (the CLI verb is
still WS-B/WS-A glue).

## Entity resolution — WS-E (`fmd/er/`)
```python
resolve(mentions, candidates, *, config: ERConfig | None = None,
        router=None, cache: AdjudicationCache | None = None,
        embedder: EmbeddingScorer | None = None,
        logger=None, transliterate=None) -> list[Resolution]
resolve_with_audit(...same args...) -> ResolutionBatch
```
Pipeline: `normalise_name` → `block` → `score_pairs` (band accept/reject/adjudicate)
→ `adjudicate` (LLM only for the ambiguous band) → `Resolution` with an
`AuditRecord` (`band, score, decided_by ∈ score|model|manual|none, features,
blocking_keys, config_version, notes`). `ResolutionBatch` also exposes
`blocking` and the model verdicts. The service is **pure**: no DB writes, no
network except an explicitly injected router; manual aliases/canonical locks win.
DuckDB/Splink/embeddings are capability-gated and fall back deterministically
(`used_fallback`, `embedding_unavailable`). `0002_er_audit_and_replay.sql`
defines `er_audit`, alias provenance spans and entity merge/blocking columns;
the service itself still performs **no DB writes**, so repository/persistence
wiring for that audit trail remains open.

## Claim schema — WS-A (`fmd/db/repositories.py:ClaimCreate`)
`extraction_id, entity_id, predicate, value, unit|None, qualifier|None, quote,
char_start, char_end, confidence=1.0, observed_at|None, method ∈ extracted|derived|
modelled|user_provided`. Append-only: use `ClaimsRepo.supersede(old,new)`. Span indexes
`extractions.text`; chain = claim→extraction→snapshot→URL+fetched_at+sha256
(`ClaimsRepo.evidence_page`).

## Hard filters — WS-F (`fmd/filters/`)
```python
FilterResult = Literal["pass", "fail", "unknown"]

@dataclass(frozen=True)
class FilterDecision:
    filter_id: str
    result: FilterResult
    claim_id: int | str | None = None
    reason: str = ""
    observed_at: datetime | None = None
    stale: bool = False
    detail: Mapping[str, Any] = {}
    def as_dict(self) -> dict[str, Any]: ...
```
`evaluate(filter_id, programme, profile, *, context=None) -> FilterDecision`;
`evaluate_all(programme, profile, *, context=None, filter_ids=None) -> FilterSummary`
(`decisions`, `order`, `counts`, `excluded`, `unverified`, `as_dict()`).
Implementations: `credential_level, recognition_status, eligibility, language,
exam_registration_open, exam_travel, intake_year_available, budget_ceiling`,
registered in `IMPLEMENTATIONS` and matched to `config/weights.yaml:filters`.
`FilterContext.from_registry` supplies TTL/currency/exam tables.
Cutoffs only — never weighted penalties. `unknown` is never silently `pass`.

## Scorer — WS-F (`fmd/score/`)
`score_programme(entity_id, programme, profile, claims, *, config=None, cost=None,
cost_rules=None, peer_prior=None, filter_decisions=(), insight_candidates=(),
now=None) -> ScoreResult`. Binding persisted output is `ScoreCreate`:
`entity_id, scorer_version, weights_version, claims_hash, sub_scores, evidence_tier ∈
well_evidenced|partial|thin|insufficient, composite|composite_lo|composite_hi|None,
computed_at|None`; persist with `fmd.score.persist.persist_score` (writes the
score row plus its `insight_sets` row).
`SubScore` = `{point, lo, hi, n_claims, n_sources, evidence_tier, measured,
prior_adjusted, components, unknown_reason, modelled_share, widened,
shrinkage_inactive}`. `shrinkage_inactive=True` means no peer prior was applied
(`prior_adjusted == measured`); it is an honest flag, not a fallback score.
UNKNOWN → `[0,100]`, never 0/50, and withholds the composite point.
Bump versions in `config/weights.yaml` on change. LLMs never emit scores.

## Cost model — WS-F (`fmd/cost/`)
`compute(CostInputs, CostRules) -> CostSnapshot`; `CostSnapshot.render()` returns
`model_dump(mode="json")` — JSON-ready, no downstream recomputation. Snapshot
carries `scenarios` (hostel/independent — **non-additive**, never summed),
`tcoa`, `tcoa_first_year`, `net_price`, `loan_principal`, `emi`, `dsr25`,
`affordability_gap`, `fx_band_pct`, `methods`, `assumptions`,
`assumptions_flags`, `display`. `loan_terms_from_config` /
`student_work_from_config` read `config/cost_rules.yaml`; an assumed loan rate is
flagged (`assumed=True`), never silently passed off as extracted.
`compute` raises `CostModelError`/`MissingFXRate`; callers must surface
"unavailable" rather than inventing a TCOA.

## API surface — WS-F engine, WS-A reads, WS-G review (`fmd/api.py`)
Envelope on every response: `scorer_version, weights_version, claims_hash, generated_at`.
Reads (WS-A): `GET /api/health`, `GET /api/sources`, `GET /api/programmes`,
`GET /api/programmes/{id}`, `GET /api/programmes/{id}/evidence`.
Engine (WS-F), profile in the **request body**, never persisted:
- `POST /api/rankings` `{profile, weights?, limit}` → ranked rows with filters,
  sub-scores, composite band, insights, cost.
- `POST /api/exams` `{profile, limit?}` → per-exam status `open|closed|unverified`,
  TTL staleness, travel-blocked by profile.
- `POST /api/exclusions` `{profile, limit?}` → fail/unknown histograms plus the
  excluded rows and deciding claim ids.
- `GET /api/programmes/{id}/cost?living=hostel|independent` → rendered snapshot or
  `available=false` with a reason.
- `GET /api/compare?ids=1,2` → Pareto matrix + pairwise comparisons.
- `POST /api/scoring/rerun` → admin-token write path.
Review (WS-G) is the **only** 501 surface: `GET /api/review/queue`,
`POST /api/review/claims/{id}` return `not_implemented` with `workstream: WS-G`.
Missing DB ⇒ 503 `database_unavailable`; invalid weights/living/ids ⇒ 422.

## Model router — WS-A (`fmd/llm/router.py`, `fmd/llm/providers.py`, `fmd/llm/doctor.py`)
`ModelRouter(models=None, *, registry=None, sink=None, daily_spent_usd=0.0, providers=None)`;
`.call(purpose, schema, system, user, input_ref) -> BaseModel`; `.ping() -> list[dict]`;
`.doctor(live=False, transport=None) -> dict` (per-profile diagnostics).
Cache key `(purpose, prompt_sha256, input_sha256, model)`. Providers are selected by
`kind`: `openai_compatible` (chat completions + `json_schema`), `anthropic` (Messages
API, guided JSON via a single `respond` tool), `gemini` (`generateContent` with
`responseSchema`), and `replay` (offline fixtures). `build_provider(name, config,
transport=...)` is the factory; tests inject `httpx.MockTransport`, never a network.
Profiles and providers are env-switchable in `config/models.yaml`; each declares
`provider_env`/`model_env` (providers also `base_url_env`/`api_key_env`,
`requires_api_key`, `supports_guided_json`) so diagnostics name exact variables.
`required: true` profiles block a run; optional ones (`hard_pdf`, `ocr_vlm`) raise
`ProviderError(reason="vision_unavailable")` per document, so text PDFs still extract.
Guided-JSON fallback: when `profile.guided_json` is false or the provider does not
support it, the router drops native schema enforcement, appends `json_instruction(schema)`
to the system prompt, and still validates the parsed object against `schema`.
CI uses `MockProvider` / `ReplayProvider` (or `fmd.extract.pipeline.build_router(mock=True)`);
no network or paid model. `fmd llm doctor [--live] [--json]` prints provider, base-URL
resolution, key *presence* (never values), model, guided-JSON expectation, reachability
(only with `--live`) and an exact remediation line; exit 0 iff every required profile is
usable. `fmd llm-ping` / `fmd llm ping` remain aliases without `--live`.

## Repositories — WS-A (`fmd/db/repositories.py`) (each takes `Database`)
Countries `upsert,get,list` · Sources `upsert,get,list` · SourceExclusions `upsert` ·
Entities `create,get,find,list` · Aliases `create,list_for_entity` · Documents
`upsert,get,get_by_url,list_by_domain` · Snapshots `create,get,latest_for_document,
list_for_document` · Extractions `create,get,list_for_snapshot` · Claims `create,get,
list_for_entity,evidence_page,supersede,count_for_entity` · Programmes `create,get,
list_for_institution,list,count,facets` · Scores `create,latest_for_entity,
list_for_entity` · Exams `upsert,get,list,upsert_window,list_windows` · CostItems
`create,list_for_programme` · FinancingOptions `create,list` · LLMCalls `record,
list_for_purpose,spend_since` · CrawlFrontier `enqueue,get,list_due,set_state,
count_by_state,set_next_allowed_at,bump_attempts,count_for_domain_since,
update_yield,yield_for_domain,record_tier,tier_stats,quarantine,is_quarantined,
get_politeness,upsert_politeness`.

`enqueue` is insert/dedupe only (keeps the earlier `next_allowed_at`); retry
backoff must call `set_next_allowed_at`, which sets the time unconditionally.
`fmd/frontier/**` holds no SQL: it calls these methods only.
