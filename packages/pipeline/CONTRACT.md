# WS-A interface contract (derived from code, not the brief)

Other workstreams implement against these names/types. Schema changes need a new
`db/migrations/NNNN_*.sql`; never edit `0001_init.sql`.
## SourceAdapter / DocumentRef — `fmd/adapters/__init__.py`
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

## Evidence store — `fmd/store/snapshot_store.py`
`SnapshotStore(root)` → `put(bytes)->sha256`, `get(sha256)->bytes`, `verify(sha256)->bool`,
`exists(sha256)->bool`, `path_for(sha256)->Path`; atomic, content-addressed
(`objects/<2hex>/<sha256>`), refuses to overwrite corrupt evidence. DB row via
`SnapshotsRepo.create`.
## Extractor protocol — `fmd/extract/` (WS-D; stub raises today)
Signature is not yet fixed; the binding artefact is the validated `ClaimCreate` row.
Reads snapshot/extraction text, yields claims whose guided-JSON fields are
`value/unit/quote/char_start/char_end/confidence`, nullable. Reject (never fix) on span
mismatch, missing unit, missing rate denominator, or out-of-band value.
## Claim schema — `fmd/db/repositories.py:ClaimCreate`
`extraction_id, entity_id, predicate, value, unit|None, qualifier|None, quote,
char_start, char_end, confidence=1.0, observed_at|None, method ∈ extracted|derived|
modelled|user_provided`. Append-only: use `ClaimsRepo.supersede(old,new)`. Span indexes
`extractions.text`; chain = claim→extraction→snapshot→URL+fetched_at+sha256
(`ClaimsRepo.evidence_page`).
## Hard filters — `fmd/filters/__init__.py` (WS-F; stubs raise today)
`FilterResult = Literal["pass","fail","unknown"]`; each `def <id>(programme: Any,
profile: Any) -> FilterResult`. IDs must match `config/weights.yaml:filters`:
`credential_level, recognition_status, eligibility, language, exam_registration_open,
exam_travel, intake_year_available, budget_ceiling`, registered in `IMPLEMENTATIONS`.
Cutoffs only — never weighted penalties.
## Scorer input/output — `fmd/score/` (WS-F; stub raises today)
`score_programme(*args, **kwargs)` is a placeholder; binding output is `ScoreCreate`:
`entity_id, scorer_version, weights_version, claims_hash, sub_scores, evidence_tier ∈
well_evidenced|partial|thin|insufficient, composite|composite_lo|composite_hi|None,
computed_at|None`, persisted via `ScoresRepo.create`. Sub-score =
`{point, lo, hi, n_claims, n_sources, evidence_tier}`; UNKNOWN → `[0,100]`, never 0/50.
Bump versions in `config/weights.yaml` on change. LLMs never emit scores.
## Model router — `fmd/llm/router.py`
`ModelRouter(models=None, *, registry=None, sink=None, daily_spent_usd=0.0, providers=None)`;
`.call(purpose, schema, system, user, input_ref) -> BaseModel`; `.ping() -> list[dict]`.
Cache key `(purpose, prompt_sha256, input_sha256, model)`. CI uses `MockProvider` /
`ReplayProvider`; no network or paid model.
## Repositories — `fmd/db/repositories.py` (each takes `Database`)
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
