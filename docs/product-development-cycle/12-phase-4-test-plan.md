# 12 — Phase 4 test plan

Companion to brief §19 and `10-inference-and-model-selection.md` §7. Phase 4
starts from `07-workstream-status.md`: **0 claims, 0 programmes, 0 scores**.
Every gate names the artefact it produces and what it does **not** prove.

Non-negotiables: CI uses `--mock` + committed fixtures (no network); no
fabricated ground truth; UNKNOWN stays UNKNOWN; a gate that passes on an empty
corpus is marked vacuous, not green.

## 1. Golden-set expansion protocol

Target 120–150 programmes, stratified by state, ownership, recognition status
and **data thinness**. Phase-4 exit first needs ≥160 verified rows across the 20
bake-off documents; programme-level expansion follows.

1. **Use a document that exists in the repo** — a committed fixture, or a
   fetched snapshot with `snapshot_id` (URL + `fetched_at` + sha256). An
   unfetched URL is not a source.
2. **Copy the quote character-for-character** into
   `data/golden/programmes/<id>.yaml`, with `printed_value`/`printed_unit` when
   normalisation changes the form. Set `fixture:` *or* `source_url` +
   `retrieved_at`; never guess either.
3. **Second person adjudicates** value, unit and span; disagreements go in
   `notes`, not silently resolved. Record curator initials. This is the
   lightweight form of the two-annotator rule; α is only reportable after ≥30
   double-labelled rows.
4. **`verified` only if the loader finds the quote verbatim.** Silent document →
   `expectation: abstain` + `absent_terms`; must-reject → `expectation: reject` +
   `reason`. Otherwise `status: unverified`, `value: null`, plus a note. Keep
   doubtful rows unverified: salary without median/Q1, rate without denominator,
   span crossing a cell boundary. No "obvious value" exceptions.
5. **Run `make eval` before committing** — the loader fails on a non-verbatim
   quote, so a typo cannot become truth. Update verified/unverified counts in
   `08-eval-methodology.md` and here.

**Does not prove:** source correctness/currency, representativeness, or that one
curator generalises. It proves traceability to a committed document at a
resolvable span.

## 2. Twenty-document bake-off

Follow `10-inference-and-model-selection.md` §7.

1. **Freeze first.** Copy 20 PDFs to `/data/bakeoff/docs`; commit a
   `manifest.sha256`, not the PDFs. Strata: 6 born-digital text, 6 scanned/mixed
   (≥3 Devanagari, one low-DPI, one skewed), 4 table-heavy (merged cells +
   footnotes), 4 adversarial (repeated numbers, column order, absent fee,
   internship-as-employment, rate without denominator). Withhold 5 from prompt
   iteration.
2. **Ground truth before candidates.** ≥8 verified fields/document, ≥160 rows,
   same schema and loader as `data/golden`; real PDFs use `source_url` +
   `retrieved_at` and are re-checked against extracted text. A zero-verified run
   is refused.
3. **Pin everything.** Production family prompt + `prompt_version`, temperature
   0, guided JSON, identical decoding per candidate. One counted schema retry; no
   silent repair, no per-candidate prompt tuning.
4. **Hard gates** (eliminate, never average): span-exact precision ≥98%,
   catastrophic rate ≤0.5%, null precision ≥95%. Report Wilson 95% CIs; report
   schema-valid rate, null recall, cost/page, p50/p95 separately.
5. **Decide:** eliminate failures; rank survivors by cost/page then p95; if the
   top two are within 2 points, declare no winner and keep the incumbent. If
   none pass, expand the corpus — do not lower gates.
6. **Publish** `report.json`, per-field table, prompt/input hashes. Only this
   report justifies changing `config/models.yaml`.

**Does not prove:** behaviour outside the strata, long-run drift, or that a
20-doc winner holds on 120–150 programmes. Twenty documents cannot resolve <2
points.

## 3. Extraction and verification gates
Per document: acquire → plan/family → extract → verify → persist.

| Gate | Pass condition | Failure behaviour |
|---|---|---|
| Schema | Guided JSON validates; every field has `value/unit/quote/char_start/char_end/confidence` | One retry on schema violation; second failure fails the document; no silent repair |
| Span | `text[start:end] == quote`; 100% of `extracted`/`derived` claims | Drop + count; one unresolved published number fails the run |
| Units | Normalises (`percent→ratio`, `inr→INR`); rate has denominator in `qualifier_json` | Reject with reason |
| Semantic | No computed/converted values; internship ≠ employment; no score/recommendation | Reject and flag the extractor |
| Provenance | Coverage ≥1.0; `modelled` excluded and counted; zero claims reports `vacuous` | Non-zero `make eval` exit |
| Verification | Quote-only pass, value not shown; only `supported` publishes | Others stay unpublished; missing verdict leaves `unverified` |
| Repair | Single field, flagged claims only; failed repair rejects | Never rewrite the claim |

**Does not prove:** that a supported claim maps to the right programme field
(e.g. a merged header), that OCR recovered every row, or that the verifier is
calibrated. Only human spot-checks settle those.

## 4. End-to-end acceptance script
Run on a clean host with a real corpus; record outputs in
`reports/phase4-acceptance.md`.

```bash
docker compose -f infra/docker-compose.yml down -v
docker compose -f infra/docker-compose.yml up -d --build
curl -sf localhost:8000/api/health                       # status ok
make migrate && make seed
psql "$DATABASE_URL" -c "select count(*) from sources where enabled"
psql "$DATABASE_URL" -c "select count(*) from countries"
make crawl ARGS='--limit 50'
make extract ARGS='--limit 20'
make verify && make score
psql "$DATABASE_URL" -c "select count(*) from programmes"
psql "$DATABASE_URL" -c "select verification_status, count(*) from claims group by 1"
make eval                    # exit 0, provenance 1.0, vacuous: false
cd apps/web && pnpm test     # 45+ pass, no 501 on a data screen
```

Manual UI proof (30 min): profile with only nationality/intake/field → ranked
shortlist with UNKNOWN; click 5 numbers → quote + URL + `fetched_at`; compare
shows "not comparable"; exclusion shows deciding claim + filter id.

Acceptance: all commands exit 0; `programmes > 0`; `claims > 0`; provenance 1.0
non-vacuous; 7 screens with real data; no number without a quote. **A DB-count
assertion that passes at 0 is a failure.** Does not prove the shortlist is good
or that coverage is complete — only that the pipeline runs and invariants hold.

## 5. Small user test with non-expert families
**Participants:** 5–8 parents/students new to the tool, mixed education/English
comfort, ≥2 without a science background. Scripted profile only; no real
personal data; nothing server-side.

**Protocol (45 min each):** (1) intro: "Evidence tool, not a counsellor; it may
say unknown"; (2) build a profile with one field unknown; (3) open the top 3
programmes and find fee and placement evidence; (4) think aloud: why was
programme X excluded? (5) compare two and identify which is *not comparable* and
why; (6) debrief: what did you trust/not believe, what would you check elsewhere?

**Observe:** time to first correct interpretation; whether they open a quote;
UNKNOWN read as "bad" vs "unknown"; internship mistaken for a job; "which is
best" handled as a band; where reading stops.

**Failure (any one fails the round):** a modelled number is read as measured;
UNKNOWN is read as a low score; exclusion is read as "bad" rather than
"unmet/unknown"; a fee is attributed to the wrong programme; no one finds the
source URL; a decision rests on one headline number. Keep verbatim quotes, not
scores.

**Does not prove:** demand, willingness to pay, or outcomes. It tests
comprehension and trust recovery; n=8 cannot support a quantitative claim.

## 6. Phase-4 exit checklist
- [ ] `make up` healthy; `make test` green offline; ruff/eslint clean.
- [ ] ≥300 institutions, ≥150 MLT programmes, recognition chain resolvable.
- [ ] 20-doc report with hard gates; model choice justified or deferred.
- [ ] ≥160 verified bake-off rows; golden counts updated (unverified never scored).
- [ ] Real claims + verification: provenance 1.0 non-vacuous; 20 spot-checked.
- [ ] Scores on real programmes; UNKNOWN propagation and intervals shown.
- [ ] 7 screens on real data; review console no longer 501.
- [ ] User test completed with failure log and fixes or documented refusals.
