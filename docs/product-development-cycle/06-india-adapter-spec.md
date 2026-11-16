# 06 — India Adapter Specification (build contract for WS-C)

**Status:** specification only. No code, no config edits, no commits.
**Author:** WS-C1. **Verification date:** 2026-11-16 (all fetches performed from a single Linux host, `curl`, no browser, no proxies).
**Consumed by:** WS-C (`fmd/adapters/*`), WS-B (fetch ladder, frontier, robots), WS-D (PDF tiers), WS-A (config loader).

This document is normative for adapter behaviour. Where it disagrees with `config/sources.yaml`, §3 lists the correction WS-C/WS-A must make; until that edit lands, the YAML value is wrong and the adapter must follow this document, not the YAML.

---

## 0. How to read this

### 0.1 Evidence markers

| Marker | Meaning |
|---|---|
| ✅ | Fetched and inspected today (2026-11-16). Status code, byte size, and extracted content are quoted below. |
| ⚠ | Could **not** be confirmed today: DNS failure, timeout, WAF block, or a claim inferred from a page I did not open. A human must check before WS-C encodes it as a fixture. |
| ⛔ | Confirmed **not buildable** under the invariants (login wall, CAPTCHA, robots disallow, or the resource does not exist). Adapter must abstain and emit `unknown`. |

Every URL in this document was read off a page I actually fetched, or is a request I actually issued. **No URL is invented.** Where a URL is a *pattern* derived from observed siblings, it is labelled `pattern:` and marked ⚠.

### 0.2 Why this matters more than usual

Three of the `base_url` values in `config/sources.yaml` are dead or misleading today (UGC `oldpdf/`, NAAC `http://`, NABH `nabh.co.in`), and several "portals" are WordPress shells where the payload is a single PDF or a JSON XHR. Building from the YAML without this document produces adapters that silently fetch 200-status error pages — UGC in particular returns **HTTP 200 with a 51,908-byte error shell** for every unknown path. A 200 is not evidence of a document anywhere on this list.

---

## 1. Verification log (what was actually fetched)

| Host | Result today | Note |
|---|---|---|
| `www.ugc.gov.in` | ✅ 200 | `Getuniversity_details` JSON XHR works; 404s return 200 error shell; DNS failed intermittently late in the session |
| `aishe.gov.in` | ✅ 200 | WordPress, robots allows all except `/wp-admin/` |
| `he.nic.in` | ✅ 200 | Serves AISHE Excel at a **static path**; TLS chain incomplete (needs system CA fix or `verify` override policy) |
| `dashboard.aishe.gov.in` | ✅ 200 | Angular SPA; API base not identified ⚠ |
| `ncahp.abdm.gov.in` | ✅ 200 | Static-ish ASP site, 5 content hubs; robots.txt 404 (absent) |
| `ahir.abdm.gov.in` | ✅ 200 | Next.js enrolment portal, login-walled ⛔ for data |
| `naac.gov.in` | ✅ 200 **only with `-k`** | `http://` times out; `https://` has incomplete cert chain |
| `www.nirfindia.org` | ✅ 200 **only with a browser UA** | Default `curl` UA → CloudFront 403 |
| `nablwp.qci.org.in` | ✅ 200, but ⛔ | `robots.txt` = `Disallow: /` |
| `nabh.co.in` | ⛔ | 302 → `ww38.nabh.co.in` → parkingcrew DNS (`185.53.179.200`) |
| `nabh.co` | ✅ 200 | Current official site; AJAX org directory works |
| `mahafra.org`, `ay25-26.mahafraportal.org` | ✅ 200 | React shell + PHP form portal |
| `paramedical.mponline.gov.in` | ✅ 200 | MP paramedical counselling entry point |
| `www.vidyalakshmi.co.in` | ⚠ timeout | Unreachable from this network/geo |
| `pmvidyalaxmi.co.in` | ⛔ WAF | F5 "Request Rejected" + support ID, all paths |
| `www.knruhs.telangana.gov.in` | ✅ 200 / ⚠ robots | Pages fine; `robots.txt` timed out twice |

---

## 2. Cross-cutting adapter rules

These bind every adapter; per-source sections only record deviations.

1. **Documents, not pages.** `SourceAdapter.fetch()` yields `DocumentRef` only. No parsing, no claim construction, no scoring (AGENTS.md).
2. **`robots_decision` is recorded per URL** before fetch (protego, project UA with contact). A `Disallow` match yields no document and a registry-visible skip reason, never a silent drop.
3. **Registrable domain, not host.** `tldextract` with the PSL's `ac.in`/`gov.in`/`nic.in` handling. `cetcell.mahacet.org` and `mahacet.org` are one budget; `cdnbbsr.s3waas.gov.in` is one budget shared by AISHE, OJEE and dozens of NIC sites — **this is the single most likely place to accidentally exceed politeness**, because a "per-source" budget would fan out to hundreds of hosts under one CDN.
4. **Default 1 rps + 300 ms jitter; `Crawl-delay` and `Retry-After` win.** No parallel connections to the same registrable domain. Nothing in this document was observed to require faster.
5. **Tier escalation only on block signals** (403/429/503/`cf-mitigated`). Escalating on a 200 error shell (UGC) is a bug: detect that case by content, not status.
6. **Drop-on-ingest fields.** UGC's university JSON includes `user_password`, `user_username`, `created_Ip`, `created_Mac`, `updated_Ip`, `updated_Mac`, `Nodal_Mob`, `Nodal_Email`, `Nodal_Phn`. These are **never** persisted into `extractions.text` or `structure_json`, never logged, never exported. Adapter must whitelist keys explicitly (§4.1.1).
7. **Conditional re-crawl everywhere.** All the WordPress targets below emit `ETag`/`Last-Modified`; honour them. UGC's JSON endpoint does not appear to — use sha256 comparison on the response body.
8. **`pdf_scan` is a format claim, not a guess.** If a PDF yields 0 characters from `pypdfium2`, set `format=pdf_scan` on the snapshot's document and route to T2/VLM (WS-D). Confirmed scanned: `dcsem.gov.in/.../l4Paymatrix.pdf` (0 chars). Confirmed born-digital: NAAC standing-committee PDFs, NCAHP MLS curriculum.

### 2.1 Per-domain posture (measured today)

| Registrable domain | robots.txt | Posture for our paths | Notes |
|---|---|---|---|
| `ugc.gov.in` | ⚠ absent (serves app HTML, 51,908 B) | Unrestricted; treat as allow | Must not treat the HTML as a robots file; validate content-type `text/plain` |
| `aishe.gov.in` | ✅ `Disallow: /wp-admin/` | Allow | |
| `nic.in` (`he.nic.in`) | ⚠ unverified | Assume allow, 1 rps | TLS chain incomplete |
| `abdm.gov.in` | ✅ 404 (absent) | Allow | |
| `gov.in` (`naac.gov.in`) | ✅ Joomla default (blocks `/administrator/`, `/components/`, …) | Allow for `/images/docs/**` | Our only NAAC path is `/images/docs/AccreditationResults/**` |
| `nirfindia.org` | ✅ 404 (absent) | Allow; **browser UA required** | |
| `qci.org.in` | ⛔ `Disallow: /` | **Do not crawl** | NABL |
| `nabh.co` | ✅ no robots served (`/robots.txt` → 301) | Allow | |
| `mahacet.org` | ✅ Googlebot/Bingbot allow-list only | ⚠ ambiguous — see §4.9.1 | Default 1 rps, conservative |
| `karnataka.gov.in` | ✅ 404 (absent) | Allow | |
| `tn.gov.in` | ✅ `Disallow: /cgi-bin/`, `/tmp/` | Allow | |
| `kerala.gov.in` | ✅ 404 (absent) | Allow | |
| `wbjeeb.nic.in` | ✅ `Disallow: /wp-admin/` | Allow | `wp-json` REST **disabled** |
| `bfuhs.ac.in` | ✅ 404 (absent) | Allow | |
| `ojee.nic.in` | ✅ `Disallow: /wp-admin/` | Allow | `wp-json` REST **disabled** |
| `ruhsraj.org` | ✅ Yoast block, `/wp-admin/` | Allow | `wp-json` returns `[]`; use HTML |
| `knruhs.telangana.gov.in` | ⚠ timed out | **Assume allow, 0.5 rps until verified** | |
| `mponline.gov.in` | ⚠ unverified | Assume allow, 0.5 rps | |
| `up.gov.in` | ⚠ unverified | Assume allow | |
| `dcsem.gov.in` | ✅ 404 (absent) | Allow | |
| `vidyalakshmi.co.in` | ⚠ unreachable | Abstain until reachable | |
| `pmvidyalaxmi.co.in` | ⛔ WAF | Abstain | |
| `mahafraportal.org` | ⚠ unverified | Assume allow, 0.5 rps (form POSTs) | |
| `kesahc.kerala.gov.in` | ✅ `Allow: /index.php` (no `Disallow`) | Allow | |
| `abvmuup.edu.in` | ⚠ unverified | Assume allow | |
| `s3waas.gov.in` (CDN) | ✅ irrelevant (no per-host robots for objects) | Allow, shared budget | Hosts AISHE/OJEE PDFs |

### 2.2 Fetch-tier expectations

| Target class | Expected tier | Rationale |
|---|---|---|
| UGC JSON, AISHE Excel, NAAC PDFs, NCAHP PDFs, MH notices, NABH AJAX | T0 `httpx` | All plain HTTP; verified today with curl |
| NIRF (`nirfindia.org`) | T0 with browser UA; T1 if UA alone fails | CloudFront filters on UA |
| `he.nic.in` | T0 with CA chain fixed | Certificate issue is local trust, not a block |
| `naac.gov.in` | T0 with CA chain fixed | Same |
| `knruhs`, `mponline`, `mahafraportal` | T0, may need T1 | Slow/opaque; not observed blocking |
| `pmvidyalaxmi.co.in` | **never** — abstain | WAF reject is a block signal; do not escalate to a browser to defeat it |
| `vidyalakshmi.co.in` | abstain until reachable | |

---

## 3. Corrections required to `config/sources.yaml`

WS-C must not edit `config/`; WS-A owns it. These are the deltas, each with evidence.

| Source id | Field | Current | Required | Why |
|---|---|---|---|---|
| `ugc_consolidated_lists` | `base_url` | `https://www.ugc.gov.in/oldpdf/` | **remove or repoint** | ✅ `/oldpdf/` returns the 51,908-byte 200 error shell, not a listing; `/oldpdf/ConsolidatedListofUniversities.pdf` also 200-error-shell. No consolidated-list PDF path found today ⚠ |
| `ugc_university_list` | `format` | `html_paginated` | `api` | Payload is a JSON XHR; there is no pagination (single response, client-side DataTable). Config note "460 state / 128 deemed / 56 central / 430 private" is stale (today: 523 / 164 / 57 / 560) |
| `naac_results` | `base_url` | `http://naac.gov.in/...` | `https://naac.gov.in/index.php/en/19-quick-links/62-accreditationresults` | ✅ `http://` times out; `https://` works (with CA fix) |
| `nabh_hospitals` | `base_url` | `https://nabh.co.in/` | `https://nabh.co/` | ⛔ apex `.co.in` 302s to a parkingcrew host |
| `nabl_labs` | `enabled` | `true` | **`false`** (or `requires_permission: true`) | `robots.txt` = `Disallow: /` |
| `counselling_madhya_pradesh` | `base_url` | `null` | `https://paramedical.mponline.gov.in/` | ✅ linked from DME MP home; dedicated paramedical counselling portal |
| `counselling_uttar_pradesh` | `base_url` | `null` | `https://abvmuup.edu.in/` | ⚠ BMLT/BMLS admissions are run by Atal Bihari Vajpayee Medical University via CAHET; `dgme.up.gov.in`/`upneet.gov.in` cover MBBS/BDS only |
| `counselling_telangana_ap` | note | one row | **split into two rows** (TS = KNRUHS, AP = Dr. NTR UHS) | Two different counselling authorities, two different payloads; a single adapter cannot honour both cadences ⚠ AP URL unverified |
| `seventh_cpc_pay_matrix` | `base_url` | `.../l4Paymatrix.pdf`, notes "Level 5/6" | keep URL, **fix the note to Level 4**, and add a second source for Level 5/6 ⚠ | Filename is `l4`; `l5`/`l6`/`level5`/`level6` all 404. Document is **scanned (0 chars)** |
| `vidya_lakshmi`, `csis_subsidy`, `pm_vidyalaxmi` | `enabled` | `true` | keep but set `expect_blocked: true` | Both hosts unreachable/WAF-rejected from a datacentre IP today |
| `state_paramedical_councils` | `notes` | "6 UTs/states had none" | add `discovery: ncahp_state_council_notifications` | NCAHP publishes the constituting notification per state; it does not publish council URLs |
| `counselling_rajasthan` + `state_health_universities` | — | both `ruhsraj.org` | merge crawl budget | Same registrable domain; must not be two independent 1 rps budgets |

---

## 4. Source specifications

Legend for the per-source tables: **Entry** = the exact URL(s) to fetch first. **Nav** = how a human reaches the payload from the entry. **Discovery** = how the adapter enumerates payloads. **Fields** = raw values present. **Map** = `claims.predicate` targets. **Accept** = the fixture WS-C commits.

### 4.1 Inventory

#### 4.1.1 `ugc_university_list` — UGC recognised universities

| | |
|---|---|
| **Entry** | `https://www.ugc.gov.in/universitydetails/university?type=MuOh4z0uqRaY2k8Ag10I0g==` ✅ (200, 55,477 B) |
| **Payload** | `https://www.ugc.gov.in/universitydetails/Getuniversity_details?unitypeID=<0..4>` ✅ JSON |
| **Nav** | Page renders 5 buttons (`#btncentral`, `#btnstate`, `#btndeemed`, `#btnstprivate`, `#btnall`); `unijs.js` calls `getuniversity()` on load and on click, GETting the endpoint above with `unitypeID` from `#hdfunitypeid` |
| **Discovery** | Enumerate `unitypeID` ∈ {1,2,3,4} (typed) and {0} (all). Measured: `0`→1,304 rows, `1`→57 central, `2`→523 state, `3`→560 private, `4`→164 deemed. `0` is the union, so `0` alone is sufficient for inventory; the typed calls are cheap and give the `uni_type` cross-check |
| **XHR replay** | ✅ Fully replayable with plain `httpx`. No cookies, no CSRF token, no Referer required (tested bare). Response is ~1.2 MB for `unitypeID=0` |
| **Raw fields** | `ID, uni_name, uni_type, status, state, address, address1, address2, Zip, url, ESTD, AIshe_code, college, Type, uniportal_id` (+ ~30 contact/admin fields to drop, §2.6) |
| **Map** | `institution.name`←`uni_name`; `institution.kind`←`uni_type`; `institution.state`←`state` (trailing spaces — trim); `institution.city`←`address` (regex, low confidence); `institution.pincode`←`Zip`; `institution.website`←`url` (a **seed**, not evidence); `recognition.status`←`status` (`2(f)`, `2(f) & 12B`, `Section-III`); `institution.aishe_code`←`AIshe_code` (often null → `unknown`); `institution.established_year`←`ESTD` (mostly null) |
| **Pagination** | None. Single response. Client-side DataTables paging is irrelevant to the adapter |
| **Rate/robots** | 1 rps. robots absent. Five calls in one session succeeded with no throttling |
| **Auth** | None. No CAPTCHA |
| **Format** | `api` (JSON). No documents to OCR |
| **Cadence / staleness** | `cadence_days: 30`. No `ETag` surfaced; staleness = age of last successful snapshot with identical sha256. **Do not** infer listing changes from `updated_date` (it is null on many rows) |
| **Trust / fraud risk** | High. Risk is not fraud but **staleness and mis-typing**: `status='Section-III'` is a deemed-university status, not a 12(B) recognition; `college`/`Type` fields are internal. A university in this list is *recognised as a university*, **not** as running an MLT programme — never let this claim set `programmes.recognition_status` |
| **Accept** | `Getuniversity_details?unitypeID=1` → JSON with `m="Data filled."`, `c="success"`, `len(List) ≥ 50`, and a row whose `uni_name == "Aligarh Muslim University"` and `state == "Uttar Pradesh "`. Fixture must assert **`user_password` is absent from the normalised record** |

⚠ Open: `ESTD` and `AIshe_code` are null in every sample row I saw (4 types). Assume `unknown` until a non-null is observed.

#### 4.1.2 `ugc_consolidated_lists` — ⛔ as configured

| | |
|---|---|
| **Entry** | `https://www.ugc.gov.in/oldpdf/` → ✅ 200, 51,908 B, `text/html`, **UGC error shell** |
| **Finding** | This is not a directory listing and not a PDF index. Three candidate PDF names under `/oldpdf/` all returned the same 200 error shell. The "consolidated university PDF" this source promised could not be located today |
| **Decision** | **Disable the adapter.** The UGC JSON (§4.1.1) already gives a strictly better version of the same inventory (structured, fresher). A separate PDF snapshot adds one thing only — a *diffable* historical artefact — and that is a WS-B/archival concern, not evidence |
| **If a human finds the real PDF** | Re-enable with `format: pdf`, and record `sha256` per annual edition so appearance/disappearance diffs are provable |
| **Accept** | None until a real URL exists. Do **not** commit a fixture that asserts a 200 error shell is a document — that bakes in the bug |

#### 4.1.3 `aishe_reports` — AISHE final reports

| | |
|---|---|
| **Entry (index)** | `https://aishe.gov.in/aishe-final-report/` ✅ (200, 110,752 B, WordPress) |
| **Payload (verified)** | `https://he.nic.in/aishereport/assets/excel/<YYYY-YY>.xlsx` ✅ — `2023-24` = 784,580 B, `2022-23` = 872,079 B, both `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`, both valid xlsx |
| **Nav** | Index table row "AISHE Final Report 2023-24 (Excel)" links to `https://he.nic.in/aishereport/#/report/2023-24`; that Angular route's `downloadFile()` builds the **static** path `${base}assets/excel/${reportId}.xlsx` (read from `main.7ba4f051919de23c.js`). So: parse the index page for `#/report/<id>`, then construct the asset URL — no browser needed |
| **Discovery** | Parse `<a href="https://he.nic.in/aishereport/#/report/…">` occurrences on the index; paginate `https://aishe.gov.in/aishe-final-report/page/<n>/` ✅ (`page/2` exists) |
| **XHR replay** | ✅ Nothing to replay: the xlsx is a plain GET. The SPA adds no auth |
| **Raw fields (25 sheets)** | `1.UniNo`, `1a.InsResponse`, `2University-Specialisation`, `3Collegerange`, `4CollegeIndicator`, `5ManagementCollegeNo`, `6TotalEnr`, `6aTotalRegularEnr`, `6bTotalDistanceEnr`, `6cTotalPrivateEnr`, `6dTotalOnline`, `7UnivActwithConsUnit`, `8CollegeAct`, `9AllSAAct`, `9aSAILevel`, `10InstituionEst`, `11Programme`, `11a programme regular`, `11b ProgrammeDistance`, `12UGDisc`, `13PGDisc`, `14-15TotalEnrCategory (2)`, `16FS-countrylevel`, `17FS-statelevel`, `18FS-prog`, `19GER(2011)`, `20GPI2011`, `21TeacherCategory`, `22TeacherPost`, `22aTeacherPostUC` |
| **MLT presence** | ✅ The workbook's shared strings contain `B.M.L.T- Bachelor in Medical Lab Technology`, `D.M.L.T - Diploma in Medical Laboratory Technician`, `Medical Lab Technology`, `Allied Health Science`, `Paramedical Science`, `Medical & paramedical`. So state-level BMLT enrolment **is** obtainable |
| **Map** | `enrolment.bmlt_ug_state_total`←(`11Programme` or `12UGDisc` row where programme ≈ `B.M.L.T`, state from `17FS-statelevel`); `enrolment.ug_total_state`←`6TotalEnr`; denominators for any "share of MLT intake" claim. **Never** maps to `programmes.seats` |
| **Not available** | College-level rows, seats, cutoffs, fees. This is **aggregate** reporting, not an institution register |
| **Pagination / rate** | Index: ~5 PDF/Excel links per page, paginated. Files: 1 request each. 1 rps |
| **Robots / TLS** | `aishe.gov.in` allows all but `/wp-admin/`. `he.nic.in` TLS chain incomplete → the fetcher must ship a CA bundle containing the missing intermediate, or `make test` will fail offline |
| **Format** | `excel`, born-digital, no OCR |
| **Cadence** | `cadence_days: 365`, but real lag is ~18 months. Staleness rule: if newest `reportId` is > 24 months behind `now`, mark all AISHE-derived claims `stale=true` and keep them out of scoring inputs that need currency (seat/eligibility) |
| **Trust / fraud** | High but **self-reported and voluntary**; non-response is systematic (small private colleges under-report) — this biases any state total downward and must never be used as a college census |
| **Accept** | GET `https://he.nic.in/aishereport/assets/excel/2023-24.xlsx` → xlsx, 25 sheets, and a shared-string `B.M.L.T- Bachelor in Medical Lab Technology`. Fixture: the workbook's `xl/workbook.xml` sheet-name list + 10 rows of `11Programme` |

⚠ AISHE **Directory of Institutions** (college-level, with AISHE codes) is a separate deliverable: `https://dashboard.aishe.gov.in/hedirectory/#/hedirectory/{universityDetails}/{C|S|U}/ALL` ✅ route exists (linked from `aishe.gov.in` home). The Angular bundle references `/aishe/reports/...` and `/aishe/Institution-Management/university`, but `https://dashboard.aishe.gov.in/aishe/reports/count/university` returns 404 and `/hedirectory/aishe/reports/…` returns 404. **The API base is unidentified.** A human must open devtools once and record the real XHR. Until then: `unknown`.

#### 4.1.4 `state_health_universities` — affiliating universities

Five configured seeds. Verdict per seed:

| Seed | Status today | Colleges payload | Notes |
|---|---|---|---|
| `https://ruhsraj.org` | ✅ 200, 665,018 B (homepage carries **617 PDF links**) | `https://ruhsraj.org/affiliated-colleges/` ✅ 200, 439,746 B | WordPress + Yoast; `wp-json/wp/v2/posts` returns `[]` → **parse HTML**. Notices also on `https://admissions.ruhsraj.org/2025/...` (pattern ⚠) and legacy `old.ruhsraj.org/cms/uploads/YYYY/MM/…` ✅. Homepage contains a broken `http://localhost/ruhs/wordpress/...` link — parser must tolerate absolute-localhost hrefs |
| `https://wbuhs.ac.in` | ✅ 200, 119,253 B | ⚠ not located | WordPress; robots disallows `/wp-content/plugins` **and** has a trailing `Allow:` line (ambiguous — treat as allow) |
| `https://www.tnmgrmu.ac.in` | ✅ 200, 166,838 B | ⚠ not located, but **sitemap exists**: `https://www.tnmgrmu.ac.in/wp-sitemap.xml` ✅ 200 | robots advertises that sitemap. Crawl sitemap → filter `/affiliation\|college\|course` |
| `https://bfuhs.ac.in` | ✅ 200, 16,174 B | ⚠ | Thin landing page. Real content on `https://bfuhs.ggsmch.org` (relabelled "University Main Site") and `https://examination.bfuhsonline.ac.in/...`. `bfuhs.ac.in/index1.aspx` ✅ 200 = admissions, 0 PDFs |
| `https://www.knruhs.telangana.gov.in` | ✅ 200, 122,953 B | `https://www.knruhs.telangana.gov.in/admission-notification/` ✅ 200, 15 PDFs | WordPress; `wp-json/wp/v2/posts` ✅ works (returns pages: `/physiotherapy/`, `/nursing-sciences/` — **these are programme pages and are the college-list entry point** ⚠). robots ⚠ timed out |

| | |
|---|---|
| **Discovery mechanism** | (a) sitemap where advertised (`tnmgrmu`), (b) `/affiliated-colleges` style page where verified (`ruhsraj`), (c) SearXNG template `site:<domain> ("affiliated colleges" OR "list of colleges")`, (d) navigation crawl depth ≤ 2 from home, filtered by anchor text matching `/\b(affiliat\|college\|institution\|course\|programme)s?\b/i` |
| **XHR replay** | Not needed; all HTML + PDF |
| **Map** | `programmes.institution_id` linkage (`college → affiliated_university`); `programmes.name`; `programmes.credential_level` (BSc = degree); `programmes.duration_months`; `programmes.seats` where a seat matrix is published; `programmes.recognition_status` (affiliation is one link in the chain) |
| **Auth / CAPTCHA** | None observed on any of the five |
| **Format** | `mixed_html_pdf`. Born-digital vs scanned: **unverified** for these universities ⚠ — route to T0 first, `pdf_scan` only on 0-char |
| **Cadence / staleness** | `cadence_days: 90`. Detect change by `Last-Modified`/ETag on the college-list page **and** by hash of the parsed college set; a list that shrinks by >10% is a parser regression signal, not a fact |
| **Trust / fraud** | High for *affiliation exists*; medium for *programme runs there*; low for anything numeric. **Gaming risk:** a college may display affiliation to a university it is not currently affiliated to (lapsed affiliation is common and rarely announced). Mitigation: require the claim to come from the **university's** page, not the college's |
| **Accept** | `https://ruhsraj.org/affiliated-colleges/` → parse ≥100 named colleges with a `state == "Rajasthan"` assertion; fixture = first 30 rows (name + any course/seats columns present) |

#### 4.1.5 `ugc_autonomous_colleges`

| | |
|---|---|
| **Entry** | `https://www.ugc.gov.in/colleges/Autonomous_Colleges` ✅ (200, 57,647 B) |
| **Nav** | UGC nav → HEIs → Colleges → "Autonomous Colleges" (href `/colleges/Autonomous_Colleges`). Page is a 5-row table of **links**; row 1 "List of Autonomous Colleges" → PDF |
| **Payload** | `https://www.ugc.gov.in/pdfnews/8688705_List-of-Autonomous-Colleges.pdf` ✅ URL observed in page; ⚠ text layer unverified (UGC DNS failed on retry today). Also present: `.../9377130_Letter.pdf`, and the 2023 Regulations PDF |
| **Discovery** | Scrape the `<table>` on the page; select anchors whose text matches `/List of Autonomous Colleges/`. Link text changes per re-issue → **do not hardcode the pdfnews filename**; resolve it from the page each run |
| **Map** | `institution.autonomous_status = true`, `quality.autonomous_as_of`←page/PDF date. Feeds `academic_quality` as a boolean, nothing more |
| **Format** | `pdf` — assume born-digital until proven otherwise ⚠ |
| **Cadence / staleness** | `cadence_days: 365`. UGC reissues this list irregularly; detect by resolved filename change |
| **Trust / fraud** | High. Risk is **inversion**: absence from this list is not evidence a college is *not* autonomous (the list lags conferments by months). Absence ⇒ `unknown`, never `false` |
| **Accept** | `https://www.ugc.gov.in/colleges/Autonomous_Colleges` → exactly one anchor matching "List of Autonomous Colleges" with a resolved `*.pdf` URL and a non-empty `sha256`. Do **not** assert PDF row contents until a human confirms the text layer |

### 4.2 Regulator

#### 4.2.1 `ncahp` — National Commission for Allied and Healthcare Professions

| | |
|---|---|
| **Entry** | `https://ncahp.abdm.gov.in/` ✅ (200, 58,798 B) |
| **Hubs that matter** | `/Curriculum` ✅ (34,653 B), `/Regulations` ✅ (17,393 B), `/Download` ✅ (25,219 B), `/StateCouncil` ✅ (36,904 B), `/WhoWeRegulate`, `/DraftCurriculum` |
| **Payloads (MLT-relevant)** | `https://ncahp.abdm.gov.in/Curriculum/MLS%20Curriculum%20NCAHP%202025%20Final.pdf` ✅ (3,749,996 B, **325 pages, born-digital**, first page text: "Competency Based Curriculum for Medical Laboratory Sciences / As per NCAHP Act 2021 / APPROVED SYLLABUS 2025"). Also `/Curriculum/DMLTCurriculum.pdf`, `/Curriculum/CurriculumCorrigendum/Corrigendum-1MLS20March2026.pdf`, `/Regulations/RAHPRegulations2026.pdf`, `/Regulations/ABandCACRegulations2026.pdf`, `/documents/Rules_Gazette.pdf` |
| **Nav** | `/Curriculum` is a flat two-column table: `Course name → PDF` plus a `Notice dt …` link per course. MLS sits under category 1 "Medical Laboratory and Life Sciences" on the home page |
| **Discovery** | Parse `/Curriculum`, `/Regulations`, `/Download`, `/StateCouncil`, `/CircularOrders`, `/ImpLetters` for `href` ending `.pdf` or containing `Curriculum`. ~100 links on `/StateCouncil`, ~89 on `/Curriculum` |
| **XHR replay** | ✅ Plain HTML; PDFs served directly (URL-encoded spaces + capitals — encode with `quote(path, safe="/")`) |
| **Raw fields (from MLS PDF)** | Programme title, duration, semester structure, course outcomes, credit hours, eligibility (10+2 PCB), internship months, assessment scheme |
| **Map** | `programmes.name`←"Medical Laboratory Sciences"; `programmes.duration_months`; `programmes.internship_months`; `programmes.credential_level`; `programmes.curriculum_version`←"2025"; `requirements.eligibility`←10+2 PCB text; `requirements.medium`. Drives hard filter 2 (`recognition_status`) **only** as *the standard a programme must meet*, never as proof an institution meets it |
| **Pagination** | None |
| **Rate / robots** | 1 rps. robots absent (404) |
| **Auth / CAPTCHA** | None on the site. ⛔ The **enrolment portals are separate hosts and login-walled**: `https://ahir.abdm.gov.in/` (institutes) ✅ 200 but shows only counters — "473 Number of College Enrolled", "784 Number of Program Enrolled" — and a "Click here for Login/Enrol"; `https://ahpr.abdm.gov.in/` (professionals) ✅ 200, login |
| **Format** | `pdf`, born-digital (✅ 0-char check passed on MLS) |
| **Cadence / staleness** | `cadence_days: 30`. Corrigenda are separate PDFs, so a curriculum claim is only valid for the **highest dated** corpus version; adapter must retain `corrigendum_index`. Staleness: if the MLS curriculum PDF's sha256 has changed, every dependent `programmes.*` claim is superseded |
| **Trust / fraud** | High — primary. **Gaming risk:** colleges advertise "NCAHP approved" — NCAHP approves **curricula and (from 2026) registers professionals**; it does **not** accredit colleges in v1 (the site's own AHIR page states: "this is data collection. IT IS NOT REGISTRATION"). Any claim `recognition_status=recognised_by_NCAHP` at *institution* level must be rejected by a plausibility gate |
| **Accept** | GET the MLS PDF → 325 pages, `pypdfium2` page-1 text contains `Competency Based Curriculum for Medical Laboratory Sciences`; and `/StateCouncil` → ≥60 anchors matching `statecouncil/.*\.pdf` |

#### 4.2.2 `state_paramedical_councils`

This is the least buildable family in the registry, and the highest fraud surface.

| | |
|---|---|
| **Entry** | `https://ncahp.abdm.gov.in/StateCouncil` ✅ — the authoritative list of *which states have constituted a council*, as dated notification PDFs |
| **Observed coverage** | ✅ 72 `statecouncil/…pdf` links covering ~30 states/UTs (AN, AP, Arunachal, Assam, Bihar, Chhattisgarh, DNH+DD, Goa, Gujarat, Haryana, HP, J&K, Jharkhand, Karnataka, Kerala, MP, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Puducherry, Punjab, Rajasthan, Sikkim, TN, Telangana, Tripura, UP, Uttarakhand, WB) — each with `SAHC_Rules/` companion PDFs. **Constituting ≠ functioning** |
| **Verified council website (1 of ~30)** | `https://kesahc.kerala.gov.in` ✅ 200, 156,986 B, WordPress. Robots: `User-Agent: *` + `Allow: /index.php` (no `Disallow` ⇒ effectively allow). Its own FAQ states registration of professionals and inspection/recognition of institutions **has not commenced** ⚠ (from search snippet of `/blog`; page not opened) |
| **Nav to institution data** | None found: no `/institutions`, `/recognised`, or `/register` links on the KESAHC home page ✅ (grep returned zero). The register does not exist publicly |
| **Discovery mechanism** | Two-stage: (1) parse `/StateCouncil` for state names + notification dates → seed table `state → council_constituted_on`; (2) per state, a SearXNG query template (§4.10) resolves a candidate website, which is then **manually approved** before it enters config (`requires_manual_review: true` already set) |
| **Map** | `regulator.state_council_constituted` (date), `regulator.state_council_rules_url`; **never** `programmes.recognition_status` unless a council publishes an institution register |
| **Auth / CAPTCHA** | n/a today — because nothing is published |
| **Format** | `mixed_html_pdf`; notification PDFs are gazette scans (assume `pdf_scan` until a 0-char test says otherwise ⚠) |
| **Cadence / staleness** | `cadence_days: 90`. Notification PDFs are append-only per state; a new dated notification supersedes, it does not edit |
| **Trust / fraud** | ⛔ **Active impersonation risk, confirmed today.** Searches surface `https://statealliedandhealthcarecouncil.com` ("State Allied And Healthcare Council Lucknow / of India", claims registration under "Indian T/Act 1882" and a "registration number 46 … recognized by the Government of India") and `https://www.sahstc.com` ("State Allied And Healthcare Skill And Training Council", "Section-64(A) Indian Trust Act 1882"). **A trust registered under the Indian Trusts Act is not a statutory State Allied and Healthcare Council under the NCAHP Act 2021.** Both are ⛔ never-to-be-crawled. Adapter rule: a council domain is admissible only if (a) it appears in an NCAHP notification, or (b) it is on a `*.gov.in`/`*.nic.in`/`*.ac.in` host, or (c) a human added it to config with a source URL |
| **Accept** | `https://ncahp.abdm.gov.in/StateCouncil` → extract `{state, notification_date, pdf_url}` for ≥30 states; assert Kerala and Maharashtra present, and that **no** extracted `pdf_url` host is outside `ncahp.abdm.gov.in`. Second fixture: `https://kesahc.kerala.gov.in/` → 200, title contains `KESAHC` or `Kerala State Allied and Healthcare` |

### 4.3 Quality

#### 4.3.1 `naac_results` — NAAC accreditation results

| | |
|---|---|
| **Entry** | `https://naac.gov.in/index.php/en/19-quick-links/62-accreditationresults` ✅ (200, **556,679 B**, Joomla) |
| **Nav** | One giant `<table>` (316,724 chars) of Standing-Committee blocks; each block has per-cycle rows (`1st Cycle`, `2nd Cycle`, …) plus appeals lists |
| **Payload pattern** | `https://naac.gov.in/images/docs/AccreditationResults/<n>SC/<file>.pdf` — ✅ 1,539 PDF anchors, ✅ SC numbers 67…265 (198 distinct), filenames irregular (`SC_265_Teacher_Education.pdf`, `SC_258_Health_Sciences.pdf`, `SC_221_Health_Sciences_R.pdf`, `1781SC_Health_science.pdf`, `63-SC-Cycle-1-HS.pdf`) |
| **Discovery** | Parse the single page for `a[href$=".pdf"]` whose anchor text matches `/health scien\|1st cycle\|2nd cycle\|CGPA/ i`. **Do not pattern-match filenames** — the naming is not consistent across years. Classify by anchor text + `<td>` context |
| **XHR replay** | ✅ Plain GET. No form, no search UI, no token |
| **Raw fields (in PDF)** | ✅ from `SC_258_Health_Sciences.pdf` (523,125 B, 1 page): `Sl. No.`, `AISHE Code` (e.g. `U-0523`), `Name of the HEI` ("KING GEORGE'S MEDICAL UNIVERSITY, LUCKNOW"), `(Re-assessment)` marker, `State` ("UTTAR PRADESH"), `CGPA` ("3.67"), `Grade` ("A++"), plus the SC meeting number and date in the header ("258th Meeting of the Standing Committee (8th August 2025)") |
| **Map** | `quality.naac_cgpa`, `quality.naac_grade`, `quality.naac_cycle`, `quality.naac_valid_from`←SC date, `quality.naac_valid_until`←date+5y (derived ⇒ `method='derived'`), `entity.aishe_code`←`AISHE Code` (**join key to AISHE and to college identity**), `institution.state` |
| **Pagination** | None on the results page (one long page). Pagination risk is **file count**, not pages: 1,539 PDFs ≈ 1,539 requests ⇒ must be a budgeted backfill, not a single crawl |
| **Rate / robots** | 1 rps. Joomla robots disallows `/components/`, `/administrator/`, etc.; `/images/docs/**` is allowed. **TLS chain incomplete** — `curl` fails without `-k`; the fetcher must trust the missing intermediate, not disable verification globally |
| **Auth / CAPTCHA** | None |
| **Format** | `pdf`, **born-digital** ✅ (text extracted cleanly, 1 page). Tabular but simple; T0 suffices. Older `*_HSC.pdf` files (2005-era) are ⚠ possibly scans |
| **Cadence / staleness** | `cadence_days: 30`. New SC ≈ 1–2/month. Staleness: a grade is valid for 5 years; adapter must record SC date and compute validity, and must **not** overwrite an older grade — supersede |
| **Trust / fraud** | High (primary). **Gaming risk:** colleges quote a *previous* cycle's grade, or an `A++` from a `Teacher Education` block rather than the institution's own. Also `_R`/`_Rev` files are revisions of the same SC — ingesting both without a supersede rule double-counts. Rule: within one SC number, `_R`/`_Rev`/`CGPA_and_Grade_Change` wins over the base file |
| **Accept** | GET `https://naac.gov.in/images/docs/AccreditationResults/258SC/SC_258_Health_Sciences.pdf` → 1 page; text contains `KING GEORGE'S MEDICAL UNIVERSITY`, `U-0523`, `3.67`, `A++`, `258th Meeting of the Standing Committee`. Fixture stores the PDF (523,125 B) + the extracted 5 fields with spans |

⚠ `http://naac.gov.in/...` (the YAML value) **times out**; only `https://` works.

#### 4.3.2 `nirf_rankings`

| | |
|---|---|
| **Entry** | `https://www.nirfindia.org/` ✅ 200 (30,522 B) **with a browser UA**; default `curl` UA → CloudFront **403** (`x-cache: Error from cloudfront`, `server: awselb/2.0`) |
| **Ranking pages** | `https://www.nirfindia.org/Rankings/2025/OverallRanking.html` ✅ (359,653 B), `…/MedicalRanking.html` ✅ (183,858 B), `…/UniversityRanking.html` ✅ (190,032 B) |
| **Payload pattern** | `https://www.nirfindia.org/nirfpdfcdn/2025/pdf/Medical/IR-D-N-15.pdf` ✅ (observed in page; the bare `IR-O-U-0001.pdf` guess 404s — **use only hrefs read from the page**) |
| **Nav** | Ranking page = 50 `<table>` elements (one per row, collapsible). Columns: `Institute ID`, `Name`, `City`, `State`, `Score`, `Rank`; each row has a `More Details` → per-institute PDF with `TLR/RPC/GO/OI/PERCEPTION` sub-scores |
| **Discovery** | For each category page: parse rows; the `Institute ID` (e.g. `IR-D-N-15`) is the stable key; ranking year is in the path |
| **XHR replay** | ✅ No JS needed once the UA is right. Pure HTML tables |
| **Raw fields** | `Institute ID, Name, City, State, Score, Rank` + PDF sub-scores `TLR(100), RPC(100), GO(100), OI(100), PERCEPTION(100)` |
| **Map** | `quality.nirf_rank_overall`, `quality.nirf_rank_medical`, `quality.nirf_score`, `entity.alias` (from `Name`) — **institution-level only** |
| **Explicit prohibition** | There is **no paramedical/MLT category** ✅ (categories present: Overall, University, Medical, Engineering, Management, Pharmacy, Law, etc.). Must never be attributed to a programme. The adapter should refuse to emit any `programme.*` claim |
| **Pagination** | None; one page per category-year |
| **Rate / robots** | 1 rps, browser UA, 0 CloudFront retries beyond 3. robots absent (404) |
| **Auth / CAPTCHA** | None. ⛔ NIRF **DCS** (the data-collection portal with the raw submitted data) is login-walled — correctly excluded in the YAML |
| **Format** | `html` + `pdf` (per-institute data sheets). PDF text layer ⚠ unverified |
| **Cadence / staleness** | `cadence_days: 365`, published ~August. Staleness: year-in-path is the version; never carry a rank forward without the year |
| **Trust / fraud** | High for the rank. **Gaming risk:** "NIRF-ranked" is used by colleges that appear in a *subject* table at rank 100–150, or in the `University` category via a different entity. Because NIRF is self-submitted data (DCS) re-scored, a rank is partly self-reported. Treat as `academic_quality` proxy with `evidence_tier` ≤ medium |
| **Accept** | `GET /Rankings/2025/MedicalRanking.html` with browser UA → ≥100 `<tr>`; first row `IR-D-N-15` / `All India Institute of Medical Sciences, Delhi`; and ≥1 anchor matching `nirfpdfcdn/2025/pdf/Medical/IR-D-.*\.pdf` |

### 4.4 Clinical infrastructure

#### 4.4.1 `nabl_labs` — ⛔ blocked by robots

| | |
|---|---|
| **Entry** | `https://nablwp.qci.org.in/laboratorysearchone` ✅ 200, 113,474 B |
| **Finding** | `https://nablwp.qci.org.in/robots.txt` = `User-Agent: *` / `Disallow: /` / "# Prevent all search engines from crawling any content" / "# Sitemap intentionally omitted" |
| **Mechanism (recorded, not to be built)** | ASP.NET WebForms: `__VIEWSTATE`, `__EVENTVALIDATION`; `ddlcountry` disabled-pending-state; name/state/organisation filters; results are postbacks |
| **Decision** | **Do not crawl.** Invariant 9 is absolute. `clinical_training` loses its NABL leg; NABH (§4.4.2) remains and is strictly better for hospitals anyway (a lab inside an accredited hospital is not what we need — we need *training sites*) |
| **What a human could do** | (a) email QCI/NABL for a data dump or permission, (b) use an officially published annual directory PDF if one exists ⚠ (not found today), (c) accept NABH-only coverage |
| **Accept** | A test asserting the adapter **refuses**: `fetch()` returns `[]` and the frontier records `robots_disallow`. That is the correct fixture |

#### 4.4.2 `nabh_hospitals`

| | |
|---|---|
| **Entry** | `https://nabh.co/find-a-healthcare-organisation/` ✅ 200, 153,890 B. **Note redirect:** slug without trailing slash 301s; host `nabh.co.in` is parked ⛔ |
| **Payload 1** | `GET https://nabh.co/wp-admin/admin-ajax.php?action=get_states` ✅ 200, 645 B, JSON array (`[" Ahmedabad","Andhra Pradesh",…]` — leading spaces, city/state mixed, incl. Nepal cities) |
| **Payload 2** | `GET …?action=get_cities_by_state&state=<name>` ✅ (pattern from `my-map.js`) |
| **Payload 3** | `POST https://nabh.co/wp-admin/admin-ajax.php` with `action=get_hospitals`, `q`, `lat`, `long`, `selectState`, `page`, `specialities[]` ✅ 200, **107,350 B JSON** for `selectState=Delhi&page=1` |
| **Nav** | The page's `#hospitalSearchForm` carries hidden `action=get_hospitals`; `my-map.js:fetchHospitals(page)` POSTs FormData and injects `data.html` into `#hospitalResults`, plus `data.mapData` for the map |
| **XHR replay** | ✅ **Fully replayable with `httpx`.** No nonce, no cookie, no CAPTCHA on this path (reCAPTCHA v3 is loaded on the page but only used by `customLoginForm` / `verify-cert.js`) |
| **Raw fields** | Response HTML columns: `HCO Name`, `Address`, `Contact`, `Acc. No.`; `data.mapData[]` presumably carries lat/lng + id ⚠ (structure not fully parsed) |
| **Map** | `hospital.name`, `hospital.address`, `hospital.state`, `hospital.accreditation_no`, `hospital.accreditation_body=NABH`, `hospital.geo`; then a derived `clinical.nabh_within_<k>_km` for each institution city (`method='derived'`) |
| **Pagination** | `page` param; response returns `data.pagination` HTML. Page size ⚠ unknown (infer from `data.mapData.length` or count rows); iterate until empty or `page > N` |
| **Rate / robots** | 1 rps. robots on `nabh.co` → 301 to itself; effectively absent. WordPress `admin-ajax.php` — **do not** hammer; one page per 2 s if the response > 100 KB |
| **Auth / CAPTCHA** | None for search. Login/verify-certificate paths are CAPTCHA/TOTP — out of scope |
| **Format** | `json` (HTML fragment inside). No PDFs |
| **Cadence / staleness** | `cadence_days: 30`. Accreditation is 3-yearly; a hospital disappearing from the list should mark `hospital.accredited=false` only after two consecutive misses |
| **Trust / fraud** | High for *this hospital is accredited*. **Gaming risk:** colleges name-drop "NABH-accredited hospital tie-up" without naming it; the derived proximity claim is **`modelled`**, must carry method + confidence + interval, and must never be rendered as "has a tie-up" |
| **Accept** | `POST admin-ajax.php` `action=get_hospitals&selectState=Delhi&page=1` → JSON with key `html` containing `HCO Name` and `Acc. No.`, and > 1 hospital row. Fixture stores the 107 KB JSON + the normalised 3 hospitals |

### 4.5 Admissions — state counselling bodies (11 entries)

Common shape: a **notice listing page** (HTML) that links dated PDFs; the useful artefacts are `seat matrix`, `cutoff / last-rank`, `brochure / eligibility`, `schedule`. Nothing here is behind login; the *application* portals are, and are out of scope.

| Source id | Verified landing / listing | Payload pattern & why | Status |
|---|---|---|---|
| `counselling_maharashtra` | `https://cetcell.mahacet.org/notices/` ✅ 200, 255,718 B, **70 PDF links/page**, pagination `https://cetcell.mahacet.org/notices/page/<n>/` ✅ (links up to `page/36` observed on page 1) | WordPress; PDFs under `/wp-content/uploads/YYYY/MM/…`. Titles in Marathi + English (UTF-8). Historic paramedical CET notices present (`DPN-PHN-CET-2026`, `FirstLst-UG.pdf`) | ✅ buildable. ⚠ robots is a Googlebot/Bingbot allow-list with no matching `User-agent: *` group — conservatively 0.5 rps and record the parsing decision |
| `counselling_karnataka` | `https://cetonline.karnataka.gov.in/kea` ✅ 200, 99,462 B | ASP.NET; notices on the home page as `…/keawebentry456/…/*.pdf`. Allied-health route today is `https://cetonline.karnataka.gov.in/kea/nursingltr2026` ("PB B.Sc, B.P.T Lateral, B.Sc AHS Lateral - 2026") ✅ link observed | ⚠ **open question:** fresh (non-lateral) BSc MLT counselling route in KA not identified today. Human must confirm whether it runs via KEA UGCET, via KNRUHS-equivalent, or college-level |
| `counselling_tamil_nadu` | `https://www.tnhealth.tn.gov.in/notifications.php` ✅ 200, 8,224 B | `<ul id="ticker01">` of `<li><a href>` — a simple ticker, no table. Links may be relative to `/` | ✅ buildable (low volume). Also `https://www.tnhealth.tn.gov.in/` ✅ 200 |
| `counselling_kerala` | `https://lbscentre.kerala.gov.in/` ✅ 200, 56,744 B; `https://lbscentre.kerala.gov.in/orders` ✅ 200, 29,930 B | Order/GO listings; `…/articles/view/<slug>/<n>` pattern; Centralized Allotments described at `/articles/view/Centralized-Allotments/2` ✅ link observed. `wp-json` N/A (custom CI app) | ✅ buildable. ⚠ `/notifications` → 404, so don't guess paths; crawl `/orders` + `/tenders` + home |
| `counselling_west_bengal` | `https://wbjeeb.nic.in/jenpas-ug/` ✅ 200, 106,541 B, **6 PDFs** | JENPAS-UG is the WB paramedical entrance. WordPress, but `wp-json` REST **disabled** (`"REST API is disabled."`) → parse HTML | ✅ buildable |
| `counselling_punjab` | `https://bfuhs.ac.in/index1.aspx` ✅ 200, 10,618 B, 0 PDFs; `https://examination.bfuhsonline.ac.in/cos/noticesview.aspx` ✅ link observed | ASP.NET. BFUHS runs Punjab's paramedical admissions **and** is a health university (§4.1.4) — one crawler, two adapters, one budget | ⚠ notices page structure unparsed; low volume |
| `counselling_odisha` | `https://ojee.nic.in/` ✅ 200, 90,717 B, 10 PDFs | PDFs on the S3-WaaS CDN: `https://cdnbbsr.s3waas.gov.in/s36832a7b24bc06775d02b7406880b93fc/uploads/YYYY/MM/<digits>.pdf` ✅ (10 observed). WordPress, REST disabled | ✅ buildable. ⚠ `/category/notice/` → 404; use home/sitemap |
| `counselling_madhya_pradesh` | `https://paramedical.mponline.gov.in/` ✅ 200 → `/Portal/services/Paramedical/home.aspx` (98,175 B) | MPOnline ASP.NET portal; `?Counstype=`-style opaque tokens on DME pages. Listing content loaded per section | ⚠ buildable but opaque: no PDFs on the landing page. Human must identify the notices section and whether it needs a session cookie |
| `counselling_uttar_pradesh` | `https://abvmuup.edu.in/admission-paramedical.php` ✅ 200, 30,407 B; `https://upneet.gov.in` ✅ 200 (254,413 B) for MBBS/BDS | ABVMU runs **CAHET** for `BMLT/BMLS/BMRIT/BPT/…`. `upneet.gov.in` has `/pdffile/UGBrochure.pdf`, `NewandNotification.aspx` (ASP.NET) but covers NEET UG only | ⚠ **correctness issue:** pointing UP at DGME would silently capture MBBS-only data. Adapter must be seeded with ABVMU and must reject MBBS seat matrices |
| `counselling_rajasthan` | `https://ruhsraj.org/` ✅ 200, 665,018 B, **617 PDFs**; `https://admissions.ruhsraj.org/2025/public/downloads/NewsFiles/…` ✅ pattern observed | RUHS is both university and counsellor. Yoast robots; REST `[]`. Also `state_health_universities` covers the same host | ✅ buildable (high volume, needs the 2 % yield cut-off) |
| `counselling_telangana_ap` | `https://www.knruhs.telangana.gov.in/admission-notification/` ✅ 200, 95,506 B, 15 PDFs; `…/wp-content/uploads/notifications/<ts>_<NAME>.pdf` ✅ | KNRUHS = Telangana only. AP = Dr. NTR UHS ⚠ (URL unverified). Feed pages `/physiotherapy/`, `/nursing-sciences/` exist as programme pages | ⚠ **must split** into two sources |

| | |
|---|---|
| **Discovery mechanism (all)** | Notice listing page → all `a[href$=".pdf"]` + `a[href*="/uploads/"]` + `a` text matching `/(seat matrix\|cutoff\|cut-off\|last rank\|brochure\|prospectus\|notification\|schedule)/i`, with the post/notice **date** scraped from the nearest date cell or the PDF filename (`YYYYMMDD…` in KNRUHS and OJEE). Date is mandatory — an undated seat matrix is unusable for `admission_realism` |
| **XHR replay** | Not needed for 8 of 11 (HTML+PDF). MP and possibly KEA need cookie/session work — treat as T1 |
| **Raw fields in seat-matrix PDFs** | College name + code, course, category/quota, sanctioned intake, available seats, round, year |
| **Map** | `programmes.seats` (per year, per quota); `admission.cutoff_percentile` (needs the closing-rank PDF, not the seat matrix); `exams.registration_opens/closes`, `exams.sitting_dates`; `programmes.eligibility`; `programmes.medium` |
| **Pagination** | Per-site: MH `/notices/page/n`; others are single listing pages (verify per site ⚠) |
| **Rate / robots** | 1 rps everywhere; 0.5 rps on `karnataka.gov.in`, `mponline.gov.in`, `tn.gov.in`; see §2.1 |
| **Auth / CAPTCHA** | None on listings. ⛔ The **candidate login** portals (`*.mponline.gov.in/Profileloginhome`, `keaonline…`, `upneet` login, `cetcell` login) are auth-walled and out of scope |
| **Document formats** | Mostly born-digital notices; **seat matrices are frequently scans** ⚠ — must be tested per file (0-char ⇒ `pdf_scan` ⇒ T2). Marathi/Tamil/Devanagari headers ⇒ WS-D VLM audit needed |
| **Update cadence / staleness** | `cadence_days: 7` is right for the *listing*; the underlying facts change on a cycle. **`observed_at` must come from the notice date, not the fetch date** — a seat matrix fetched today but published on 2026-08-11 is 3 months stale by definition. Filter 5 (`exam_registration_open`) must use notice date + published window, never fetch time |
| **Trust / fraud** | High (state authority). **Gaming/fraud risk is misattribution, not falsification:** private colleges publish "seats" that are management-quota only, or quotas that require domicile the applicant does not have; coaching sites republish cutoffs with transcription errors. Rules: (a) quota is mandatory metadata; (b) never merge management/NRI quota into a sanctioned-intake figure; (c) a cutoff claim needs `qualifier_json.quota + category + year + round` or it is rejected (AGENTS.md data rule) |
| **Accept (per site, minimum)** | MH: `GET /notices/` → ≥50 PDF anchors + a `notices/page/2/` link. TN: `notifications.php` → ≥1 `ticker01` list item with an href. WB: `/jenpas-ug/` → ≥1 PDF. KNRUHS: `/admission-notification/` → ≥5 PDFs whose filename matches `^/wp-content/uploads/notifications/\d{14}_`. OJEE: home → ≥5 `cdnbbsr.s3waas.gov.in/.../*.pdf`. RUHS: home → ≥100 `*.pdf`. MP: `paramedical.mponline.gov.in` → 200 + title contains `Paramedical`. UP: ABVMU page contains `BMLT` or `BMLS` |

### 4.6 Fees

#### 4.6.1 `fee_regulators` (MH, RJ, KL, TN, MP)

| | |
|---|---|
| **Verified component — Maharashtra** | Portal: `https://ay25-26.mahafraportal.org/ssi_prp_24/outer.php?q=fee_search_report` ✅ 200, 13,875 B. Title: "Online Fee Approval Proposal Portal of F…". Forms: `form#home`, `#e11`, `#d`, `#b`, `name="login"` (POST to `outer.php`). Filters: `select[name=type]` (`HT`, `MEDIC`, `AGRI`), `select[name=sub_type]` (**empty; populated by JS on `type` change** ⚠), `select[name=district]` (`all` + ~36 districts). Public info site: `https://mahafra.org` ✅ 200 but only 875 B (React shell) |
| **Blocked at** | A POST with `type=MEDIC&sub_type=&district=all` returned the same 13,875 B form page (no results) — `sub_type` must be filled from the JS-populated list. **The `sub_type` vocabulary is unidentified** ⚠ — a human must open the page in a browser, pick "Medical", and record the `<option>` values once. Whether **paramedical** appears as a sub-type under `MEDIC` is unknown and is the crux of this source |
| **Other states** | ⚠ No verified portal URL today for RJ, KL, TN, MP. Kerala's Fee Regulatory Committee publishes **orders as PDFs via CEE Kerala** (`cee-kerala.org/whatsnew/file/<id>` ✅ seen in search results, not fetched) and via college mirrors — there is no standalone FRC portal. Rajasthan/TN/MP: not verified |
| **Discovery** | (a) MH: form replay once `sub_type` vocabulary is known; (b) others: SearXNG template `"<state>" fee regulatory (commission\|committee\|authority) fee order <course>` constrained to `site:gov.in`; then manual approval |
| **Map** | `cost_items.tuition` (`category=tuition`, `period=per_year`, `currency=INR`, `as_of=<order year>`), `cost_items.other_fees`; `cost.fee_order_ref` |
| **Auth / CAPTCHA** | MH form has a `login` form on the same page but the **search report** did not demand auth. No CAPTCHA seen ⚠ |
| **Format** | MH: HTML table (if reachable). Others: `pdf_scan` (per YAML) — gazette orders are almost always scans; assume T2 |
| **Cadence / staleness** | `cadence_days: 365`. Orders are per academic year ⇒ `as_of` mandatory; a 2024-25 order cannot price a 2027 intake |
| **Trust / fraud** | High where an order exists. **Coverage fraud risk is the real one:** colleges quote a "government-approved fee" that covers only the tuition head, excluding special fees, exam fees, and "development" charges that the order does not regulate. Also many orders cover **professional courses only** (MBBS/BDS/engineering) and say nothing about BSc MLT ⇒ `unknown`, never `0` and never a modelled value sourced from an MBBS order |
| **Accept** | MH: `GET …?q=fee_search_report` → 200, page contains `select[name=type]` with options `HT`, `MEDIC`, `AGRI`. Fixture asserts the three option values and that `sub_type` has exactly one option (proving it is JS-populated). **No numeric fixture until a human supplies a working POST + the MLT/paramedical sub-type** |

### 4.7 Pay anchor

#### 4.7.1 `seventh_cpc_pay_matrix`

| | |
|---|---|
| **Entry** | `https://dcsem.gov.in/resources/uploads/rti_links/l4Paymatrix.pdf` ✅ 200, 358,030 B, `application/pdf`, **1 page, 0 extractable characters ⇒ scanned** |
| **Finding 1** | The document is **Level 4**, not "Level 5/6" as the YAML note claims. MLT graduate-entry posts (Lab Technician / Lab Assistant in central hospitals) commonly map to Level 4–6, so the *anchor level matters* |
| **Finding 2** | `l5Paymatrix.pdf`, `l6Paymatrix.pdf`, `l7Paymatrix.pdf`, `level5Paymatrix.pdf`, `level6Paymatrix.pdf` → **all 404** ⚠. `dcsem.gov.in/` home lists only `hindi/scale_of_pay.html` and its English twin `english/scale_of_pay.html` — neither page yielded a matrix PDF link in the grep |
| **Discovery** | None reliable yet. Human must locate the official Level 5/6 matrix (candidates: the 7th CPC report annexures, DoE/DoPT circulars, or `dcsem.gov.in/english/scale_of_pay.html`) |
| **Map** | `salary.pay_level`, `salary.pay_matrix_cell` (`level × index`), `salary.basic_pay`, then derived `salary.gross_estimate` = basic + DA + HRA + TA (`method='modelled'`, with interval). Anchors the **government** salary band only |
| **Format** | `pdf_scan` ⇒ T2 (Docling/MinerU) or a small VLM pass. 1 page, so cost is trivial — this is a good first T2 end-to-end test |
| **Cadence / staleness** | `cadence_days: 3650` (a pay commission is a decade). Staleness: **DA rate changes twice a year** — the matrix itself is stable, but `allowances.da_rate` must be a separate source or `unknown` |
| **Trust / fraud** | High. **Modelling risk, not fraud:** the matrix gives *basic pay*. Quoting "₹56,100/month" without DA/HRA or without saying it is basic is the most common misrepresentation of government MLT pay in college brochures |
| **Accept** | GET the PDF → 358,030 B, `pypdfium2` page-1 chars **== 0** (asserting the scan classification), and T2 output containing a recognisable `Level 4` header + a numeric matrix. Fixture = the PDF + the T2 JSON, pinned by `tool_version` |

### 4.8 Financing

#### 4.8.1 `vidya_lakshmi` / `csis_subsidy`

| | |
|---|---|
| **Entry** | `https://www.vidyalakshmi.co.in/` ⚠ **connection timeout (40 s, twice)**; `https://www.vidyalakshmi.co.in/Students/resources/SubsidyScheme.html` ⚠ same; `/Users/Login` ⚠ same |
| **Interpretation** | Network-level block or geo-restriction from this host. Not a 403, not a CAPTCHA — the TCP connection never establishes. Could be the site, could be the egress path. **Unverified, not blocked-by-policy** |
| **Decision** | Adapter must be written but `enabled` only after a human confirms reachability from the deployment egress. On failure ⇒ emit nothing, log `unknown`, never fabricate scheme terms |
| **Map (when live)** | `financing_options.lender`, `.scheme`, `.rate_min/.rate_max`, `.max_amount`, `.moratorium_months`, `.tenure_months`, `.collateral_rules`, `.subsidy_rules`; `cost.student_work` stays `none` for India |
| **Auth** | Scheme comparison pages are public ✅ per YAML note (unverified today). Application flow is login-walled ⛔ and excluded |
| **Format** | `html` |
| **Cadence / staleness** | `cadence_days: 30`. Interest rates on CSIS change with repo-linked benchmarks ⇒ any rate claim older than 90 days is `stale` |
| **Trust / fraud** | High for scheme existence. **Gaming risk:** CSIS eligibility for BSc MLT as a "professional course" is **lender-dependent** (YAML already says so) ⇒ must be `conditional`, and the UI must not present subsidy as approved. Second risk: colleges quote "education loan up to ₹X" (the scheme cap) as if it were a sanction |
| **Accept** | A test that asserts **abstention**: adapter returns `[]`, records `unreachable`, and no `financing_options` row is written. When a human confirms reachability, fixture = the CSIS page HTML + parsed rate fields |

#### 4.8.2 `pm_vidyalaxmi`

| | |
|---|---|
| **Entry** | `https://pmvidyalaxmi.co.in/` ⛔ **WAF reject**: 302 → `/Page_50x.aspx?aspxerrorpath=/`, 246 B, `<title>Request Rejected</title>` + F5 support ID (`8393570473100769769`). Same for `https://www.pmvidyalaxmi.co.in/` and `/Home.aspx` |
| **Interpretation** | Active application-layer defence that rejects datacentre traffic. This is a **block signal** ⇒ per §7 of the brief, escalate to `curl_cffi` impersonation **at most once**, then quarantine. **Do not** escalate to a headful browser to defeat a WAF, and never solve anything |
| **Decision** | Treat as effectively non-existent for v1. `financing_feasibility` uses Vidya Lakshmi/bank terms + `unknown` for PM-Vidyalaxmi. If a human can retrieve the QHEI list + scheme PDF from a browser session, hand it to the adapter as a **committed fixture**, not a live crawl |
| **Map (if obtained)** | `financing_options.scheme="PM-Vidyalaxmi"`, `.max_amount`, `.rate_min/max`, `.moratorium_months`, plus the QHEI-institution flag (relevant only for top-tier institutions; most MLT colleges will not appear) |
| **Accept** | A test asserting the adapter classifies the response as `waf_blocked` and abstains — **not** a fixture of the 246-byte reject page pretending to be content |

### 4.9 Institution sites (`institution_site`) — placement-PDF discovery

This is the highest-variance adapter and the main source of `outcomes` evidence. It is `trust: low` by definition.

| | |
|---|---|
| **Where placement numbers actually live** | (1) **NAAC SSR** — the Self-Study Report submitted before a NAAC peer team visit; Criterion 5/6/7 annexures carry placement tables (`number of students placed`, `median salary`, recruiter names). (2) **AQAR** — annual, published by the college, with a "Placement and Higher Studies" table per programme. (3) **NIRF DCS submission mirrors** — colleges post their own "NIRF 2025 Data" PDFs (the DCS itself is login-walled ⛔, the college's mirror is not). (4) **Annual report / IQAC report**. (5) **Prospectus** (fees, seats, hospital tie-ups). (6) **Placement cell page** (usually marketing). |
| **Verified pattern (structure, not URL)** | NAAC results themselves are at `naac.gov.in/images/docs/AccreditationResults/...` — SSRs are **not** there ✅ (YAML note correct: "SSR/AQAR PDFs live on college sites, not here") |
| **Discovery mechanism (ordered, cheapest first)** | 1. **Sitemap**: `https://<domain>/sitemap.xml`, `/wp-sitemap.xml`, `/sitemap_index.xml` (tnmgrmu ✅ `wp-sitemap.xml`). 2. **Well-known paths** (probe, expect 404): `/naac`, `/iqac`, `/naac/ssr*`, `/nirf`, `/placement*`, `/downloads`, `/documents`. 3. **Site search** (`/?s=SSR`, `/?s=placement+report`) on WordPress hosts. 4. **SearXNG templates** (§4.10). 5. **Common Crawl index** for `*.ac.in` PDFs whose URL contains `ssr\|aqar\|nirf\|placement`. |
| **File-name heuristics that pay** | `(SSR\|AQAR\|NIRF\|placement\|annual.?report\|prospectus).*\.pdf`, `(cycle\|CYCLE)[-_ ]?[12]`, `criteria[-_ ]?[5-7]`, `DCS`, `AISHE`. Combine with a fetch of the anchor text |
| **XHR replay** | n/a (static files). But college sites are frequently on shared hosting with Cloudflare ⇒ expect tier escalation here more than anywhere else |
| **Raw fields** | Programme name, batch year, `no. of students graduated`, `no. placed`, `median / max salary`, recruiter list, higher-studies count, internship site names, hospital bed counts |
| **Map** | `outcomes.placement_rate` (**denominator mandatory** in `qualifier_json`), `outcomes.median_salary`, `outcomes.p25_salary`, `outcomes.n`, `outcomes.recruiter_count`, `clinical.tied_hospitals[]`, `clinical.beds`, `cost.equipment`, `programmes.seats` |
| **Pagination** | n/a; bounded by `max_pages_per_domain` (propose 200) and the `yield_score < 2 %` stop rule |
| **Rate / robots** | 1 rps, per registrable domain. Many `*.ac.in` hosts have no robots ⇒ allow; honour `Crawl-delay` where present. Expect a long tail of dead domains: a 404/410 is terminal **and must not trigger tier escalation** |
| **Auth / CAPTCHA** | Occasionally a college puts SSRs behind a login (rare) ⇒ abstain, record `login_walled` |
| **Format** | Predominantly `pdf`. SSRs are a **mix**: born-digital in ~50 % of cases, scanned (with a signature page) in the rest ⇒ per-file 0-char test, T2 for scans. Tables with merged cells are the known Docling failure mode (AGENTS.md G6.3) ⇒ row-count reconciliation + VLM audit |
| **Cadence / staleness** | `cadence_days: 90`. SSR is per-NAAC-cycle (5 years) — stale by construction. AQAR is annual. Staleness rule: `report_year` must be extracted; a placement claim without a year is rejected |
| **Trust / fraud** | **Low, and adversarial.** Documented gaming patterns to encode as red flags on the *claim*: (a) placement rate with no denominator ("100 % placement"); (b) "highest package" as the headline with no median/P25; (c) rounded figures (₹3,00,000, ₹4,50,000) across years; (d) recruiters named without counts; (e) offers counted as joinings; (f) programme-level table silently aggregating all streams; (g) internship stipends presented as salaries; (h) YoY jump > 2σ peers; (i) SSR copy-pasted between sister institutions (simhash catches this — reuse the snapshot dedupe) |
| **Accept** | Domain-level fixture: a committed mini-corpus of 3 real SSR/AQAR PDFs (one born-digital, one scanned, one with a merged-cell table) + an assertion that the extractor finds a placement table with a **denominator** or returns `unknown`. Plus a negative fixture: a "100 % placement" slide with no denominator ⇒ claim rejected by the validation gate |
| **Non-negotiable** | Internships are **not** employment. A "1-year internship at Apollo" line is `clinical.internship_site`, never `outcomes.placed` |

### 4.10 Discovery sources (`searxng`, `common_crawl_index`) — templates WS-C depends on

These are WS-B's, but every adapter above needs their query templates. Record them here so the templates are versioned with the spec.

| Purpose | SearXNG query template |
|---|---|
| State council website | `"<state>" ("state allied and healthcare council" OR "state paramedical council") (site:gov.in OR site:nic.in OR site:ac.in)` |
| College → university affiliation | `site:<university-domain> "<college name>"` |
| SSR/AQAR | `site:<college-domain> (SSR OR AQAR OR "self study report") filetype:pdf` |
| NIRF mirror | `site:<college-domain> NIRF filetype:pdf` |
| Seat matrix | `"<state>" "<course>" seat matrix filetype:pdf` |
| Fee order | `"<state>" fee (order OR notification) "<course>" filetype:pdf` |

Common Crawl (DuckDB over the parquet index): `url REGEXP '.*\.(pdf)$' AND url LIKE '%.ac.in/%'` plus a path filter on `ssr|aqar|nirf|placement|prospectus`. Results are **leads only** — every one must be re-fetched and snapshotted before it becomes evidence.

---

## 5. Build order — top 10 by (information gained ÷ scraping difficulty)

Score = how much of the schema it fills, per unit of adapter work. Difficulty assumes T0 HTTP only.

| Rank | Source | Info gained | Difficulty | Why this rank |
|---|---|---|---|---|
| 1 | **UGC university JSON** (§4.1.1) | Institution spine: 1,304 names, type, state, pincode, website, 12(B)/2(f) status | **Trivial** (1 JSON GET, no auth) | Unblocks entity resolution for every other adapter. Highest ratio on the list |
| 2 | **NAAC results PDFs** (§4.3.1) | CGPA + grade + cycle + **AISHE code** per institution | Low (HTML index → born-digital PDFs) | The AISHE code is the join key that makes ER cheap. Direct `academic_quality` input |
| 3 | **NCAHP corpus** (§4.2.1) | Curriculum, duration, internship months, eligibility, 2025/2026 regulations | Low (static PDFs, ~5 pages of links) | Drives hard filter 2 and the programme schema. Tiny volume, huge normative weight |
| 4 | **NABH AJAX directory** (§4.4.2) | Accredited hospital list + geo | Low-medium (one POST, pagination) | Only remaining `clinical_training` leg after NABL is blocked |
| 5 | **MH CET Cell notices** (§4.5) | Seat matrices, cutoffs, eligibility, real candidate-facing deadlines | Low (WordPress, 70 PDFs/page, 36 pages) | First real `admission_realism` + `seats` evidence; also the template every other counselling adapter copies |
| 6 | **AISHE Excel** (§4.1.3) | State-level BMLT/B.Sc MLT enrolment + denominators | Low (1 xlsx per year) | Supplies the denominator for any "share of MLT" claim and a state-level sanity band |
| 7 | **KNRUHS + RUHS + OJEE + WBJEEB notices** (§4.5) | 4 more states of seats/cutoffs | Low each (HTML+PDF) | Same adapter shape as #5; marginal cost near zero after #5 |
| 8 | **State health universities** (§4.1.4) | College↔university affiliation chains; syllabus/seat data | Medium (5 heterogeneous sites, sitemaps) | Completes the recognition chain required by hard filter 2 |
| 9 | **NIRF** (§4.3.2) | Institution-level rank/score proxy | Medium (UA-gated, big pages) | Small scoring contribution; do it after the spine exists |
| 10 | **Institution sites / SSR-AQAR** (§4.9) | Placement + salary + hospital tie-ups | **High** (long tail, scans, adversarial) | Highest *absolute* value but worst ratio — build last, on top of a working ER and PDF stack |

**Explicit non-order:** fee regulators (§4.6), pay matrix (§4.7), financing (§4.8) are cheap individually but cover little; do them when a `cost` or `financing_feasibility` number is needed for the UI, not before.

---

## 6. Cannot be built (abstain, do not work around)

| Source / aspect | Status | Reason | Adapter behaviour |
|---|---|---|---|
| **NABL accredited labs** (`nablwp.qci.org.in`) | ⛔ | `robots.txt` = `Disallow: /` | Return `[]`, log `robots_disallow`. `clinical_training` falls back to NABH only |
| **NIRF DCS bulk data** | ⛔ | Login-walled (correctly excluded in YAML) | Only college-hosted mirrors, as low-trust institution evidence |
| **NCAHP AHIR institute registry** | ⛔ for data | Login-walled ("Click here for Login/Enrol"); site itself states enrolment is *data collection, not registration* | Read the two public counters only if useful; never claim an institution is enrolled |
| **NCAHP AHPR professional registry** | ⛔ | Login-walled | n/a for v1 (G2.2 is BLOCKED in the backlog) |
| **PM-Vidyalaxmi** (`pmvidyalaxmi.co.in`) | ⛔ | WAF rejects datacentre traffic on every path (F5 `Request Rejected`) | Classify `waf_blocked`, abstain. Do not escalate to a browser to defeat it |
| **Vidya Lakshmi / CSIS** (`vidyalakshmi.co.in`) | ⚠→⛔ | TCP timeout from the test network (twice). Not a policy block, but unreachable today | Attempt once per run, then abstain; needs a human reachability check from the deploy egress |
| **UGC consolidated university PDFs** (`/oldpdf/`) | ⛔ | Path is not a listing; serves a 200 error shell. No replacement found | Disable; the JSON supersedes it |
| **UGC "fake universities" list as a hard filter** | ⚠ | Exists in UGC nav; URL not verified today | High value (it is a *cutoff*, per invariant 4) but must not be invented — a human should supply the URL |
| **State councils with no site** | ⛔ for ~29 of ~30 states | Only NCAHP constituting notifications exist; **the register does not exist**. Confirmed for Kerala: registration "not commenced" | Emit `regulator.state_council_constituted=<date>` only; `recognition_status` stays `unknown` |
| **Third-party "State Allied and Healthcare Council" sites** | ⛔ | Confirmed impersonators: `statealliedandhealthcarecouncil.com`, `sahstc.com` — private trusts under the Indian Trusts Act, not statutory councils | Domain denylist; never fetch, never cite |
| **Fee orders for paramedical courses (most states)** | ⛔/⚠ | MH sub-type vocabulary unknown; KL has no FRC portal (orders via CEE + college mirrors); RJ/TN/MP unverified. Orders that exist often cover professional courses only | `unknown` tuition unless an order explicitly names the course; never interpolate from an MBBS order |
| **7th CPC Level 5/6 matrix** | ⚠ | Only the Level 4 PDF verified; `l5/l6/level5/level6` all 404 | Use Level 4 + a modelled band, or get the correct URL from a human |
| **Fresh-entry BSc MLT counselling route in Karnataka** | ⚠ | Only lateral-entry (`nursingltr2026`) and UGCET links observed | Abstain on KA seat/cutoff until confirmed |
| **MP paramedical counselling list** | ⚠ | Landing page verified, but no PDFs and no identified notices section | Abstain until a human maps the notices section |
| **AP (Dr. NTR UHS) counselling** | ⚠ | Folded into a Telangana row; URL unverified | Split the source, mark AP `unknown` until verified |
| **Any CAPTCHA-gated portal** | ⛔ | Policy (YAML `exclusions`) | `unknown`, log, never solve |

---

## 7. Data dictionary — every pipeline field mapped to at least one source

`✅` = a verified source covers it. `⚠` = partial / needs a human. `⛔` = no source in v1 (must render `unknown`).

### 7.1 Institution

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `entities.canonical_name` | UGC JSON `uni_name` | NAAC HEI name, NIRF `Name` | ✅ |
| `kind` (institution) | UGC `uni_type` (`Central/State/Private/Deemed`) | College pages | ✅ |
| `state` | UGC `state` | NAAC `State`, counselling PDFs | ✅ |
| `city` | UGC `address` (regex) | NIRF `City` | ⚠ |
| `pincode` | UGC `Zip` | — | ✅ (partial coverage) |
| `website` | UGC `url` | SearXNG lead | ✅ (seed only) |
| `aishe_code` | NAAC `AISHE Code`, UGC `Aishe_code` | AISHE directory ⚠ | ✅ via NAAC |
| `established_year` | UGC `ESTD` (null in samples) | college page | ⚠ |
| `ownership` | derived from `uni_type` + college page | — | ⚠ |
| `autonomous_status` | UGC autonomous PDF | college page claim (low) | ✅ |
| `affiliated_university` | state health university college lists | college page | ✅/⚠ |
| `status` (12(B)/2(f)) | UGC `status` | — | ✅ |
| `fake_university` | UGC fake list | — | ⚠ URL unverified |

### 7.2 Programme

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `programmes.name` | NCAHP MLS curriculum; university college list | counselling seat matrix | ✅ |
| `credential_level` | counselling/university page (`B.Sc` vs `DMLT`) | NCAHP | ✅ |
| `duration_months` | NCAHP MLS curriculum | university syllabus | ✅ |
| `internship_months` | NCAHP MLS curriculum | SSR | ✅ |
| `seats` | state counselling seat matrix | university/prospectus | ⚠ (state-dependent) |
| `medium` | university page | counselling brochure | ⚠ |
| `recognition_status` | **chain**: UGC university status → affiliation → NCAHP curriculum → council | — | ⚠ (council leg missing in ~29 states) |
| `curriculum_version` | NCAHP (2025 + corrigenda) | — | ✅ |
| `eligibility` (10+2 PCB) | NCAHP curriculum; counselling brochure | — | ✅ |

### 7.3 Quality

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `naac_cgpa` / `grade` / `cycle` | NAAC SC PDFs | college claim (low) | ✅ |
| `naac_valid_until` | derived (SC date + 5y) | — | ✅ (`derived`) |
| `nirf_rank` / `score` | NIRF ranking pages | — | ✅ (institution-level only) |
| `autonomous_status` | UGC PDF | — | ✅ |

### 7.4 Clinical

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `nabh_nearby` | NABH AJAX + geo | — | ✅ (`modelled`) |
| `nabl_nearby` | ⛔ robots | — | ⛔ |
| `tied_hospitals[]` | institution SSR/prospectus | — | ⚠ (self-reported) |
| `hospital_beds` | SSR/prospectus | — | ⚠ |
| `internship_sites[]` | SSR/AQAR | — | ⚠ |

### 7.5 Cost

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `tuition` | fee regulator order (MH partial) | college prospectus (low) | ⚠/⛔ |
| `other_fees` | fee order | prospectus | ⚠ |
| `hostel` / `mess` | prospectus | city basket (`modelled`, ±20 %) | ⚠ |
| `equipment` | prospectus / student mentions | `modelled` | ⚠ |
| `exam_fee` | counselling brochure | exam portal ⛔ | ⚠ |

### 7.6 Admissions / exams

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `exams.name` | counselling pages (MHT-CET, JENPAS-UG, CAHET, …) | — | ✅ |
| `registration_opens/closes` | counselling notice PDFs | — | ✅ (per state, per cycle) |
| `sitting_dates` | counselling schedule PDF | — | ✅ |
| `requires_travel` | derived from exam centre policy ⚠ | — | ⚠ |
| `cutoff_percentile` | counselling cutoff/last-rank PDFs | — | ⚠ (not verified for any state today) |
| `seat_matrix` | counselling PDFs | — | ✅ (MH verified; others ⚠) |

### 7.7 Financing

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `lender`, `scheme` | Vidya Lakshmi / CSIS page | bank pages | ⛔ today (unreachable) |
| `rate_min/max`, `max_amount`, `moratorium`, `tenure` | same | — | ⛔ today |
| `subsidy_rules` (CSIS) | CSIS page | — | ⛔ today; `conditional` when obtained |
| `PM-Vidyalaxmi` terms | ⛔ WAF | — | ⛔ |

### 7.8 Outcomes

| Field | Primary | Fallback | Status |
|---|---|---|---|
| `placement_rate` (with denominator) | SSR/AQAR Criterion 5/6 | NIRF DCS mirror | ⚠ (low trust) |
| `median_salary` / `p25_salary` / `n` | SSR/AQAR | — | ⚠ (rarely published) |
| `recruiter_count` | SSR/AQAR | placement-cell page | ⚠ |
| government pay anchor | 7th CPC Level 4 (verified) / 5-6 ⚠ | — | ⚠ |
| `registration_conversion` | ⛔ (no public register) | — | ⛔ (backlog G2.2) |

### 7.9 Coverage summary

- **Solid (✅)**: institution spine, quality (NAAC/NIRF/autonomous), curriculum/eligibility, hospital proximity, state-exam windows for ~8 states, seat matrices for ≥1 state.
- **Thin (⚠)**: seats/cutoffs outside MH, affiliation chains, fee orders, salary/placement, medium of instruction, `requires_travel`.
- **Absent (⛔)**: lab (NABL) proximity, professional registration conversion, PM-Vidyalaxmi, most state fee orders, state council registers, 7th CPC Level 5/6.

---

## 8. Fixture & acceptance-test plan (what WS-C commits)

`make test` must run all of this offline (AGENTS.md). So each item below is a committed artefact + an assertion, not a live call.

| # | Fixture | Assertion |
|---|---|---|
| F1 | `ugc_university_central.json` (57 rows) | 57 rows; contains "Aligarh Muslim University"; **no `user_password` key after normalisation** |
| F2 | `naac_SC_258_Health_Sciences.pdf` + extracted JSON | `U-0523`, `3.67`, `A++`, `258th Meeting … (8th August 2025)` with resolving spans |
| F3 | `ncahp_MLS_2025.pdf` page 1 | 325 pages; page-1 text contains `Medical Laboratory Sciences` and `APPROVED SYLLABUS` |
| F4 | `ncahp_statecouncil.html` | ≥30 `{state, date, pdf_url}` rows; all hosts == `ncahp.abdm.gov.in` |
| F5 | `nabh_get_hospitals_delhi_p1.json` | keys `html`, `pagination`, `mapData`; `html` contains `Acc. No.` |
| F6 | `aishe_2023-24.xlsx` `workbook.xml` + sample rows | 25 sheets; shared string `B.M.L.T- Bachelor in Medical Lab Technology` |
| F7 | `mh_cetcell_notices.html` | ≥50 PDF anchors; a `notices/page/2/` link |
| F8 | `nirf_2025_medical.html` | ≥100 `tr`; row `IR-D-N-15` + `All India Institute of Medical Sciences, Delhi` |
| F9 | `nabl_robots.txt` | adapter returns `[]` with reason `robots_disallow` |
| F10 | `pmvidyalaxmi_reject.html` | adapter classifies `waf_blocked` and emits no claim |
| F11 | `ugc_error_shell.html` (51,908 B) | adapter raises `UgcErrorShell` and does **not** yield a document — guards against the 200-status trap |
| F12 | 3 SSR/AQAR PDFs (born-digital / scanned / merged-cell) | placement table parsed with denominator, else `unknown`; "100 % placement" with no denominator ⇒ claim rejected |
| F13 | `dcsem_l4Paymatrix.pdf` | page-1 chars == 0 ⇒ `format=pdf_scan` ⇒ routed to T2 |
| F14 | `kesahc_home.html` | 200; no `*.pdf` register link extracted (proves the register is absent) |

Harness rules: fixtures store the **raw bytes** and the sha256; assertions are on normalised records; every fixture names its `sources.yaml` id; a fixture whose source is `enabled: false` still runs (that is how the abstention tests stay honest).

---

## 9. Open questions — a human must answer these before WS-C finishes

1. **UGC consolidated PDF**: does an official consolidated university list PDF still exist? If not, confirm `ugc_consolidated_lists` should be deleted rather than fixed.
2. **UGC fake-universities list URL**: found in the site nav, not resolvable today. Hard-filter value is high.
3. **AISHE directory XHR**: open `dashboard.aishe.gov.in/hedirectory/#/hedirectory/universityDetails/S/ALL` in devtools, record the request that fills the table. This is the only path to a college-level AISHE register.
4. **MH FRA `sub_type` vocabulary**: does "paramedical" / "B.Sc MLT" appear under `type=MEDIC`, `HT`, or a separate type? One browser session answers the whole fee question for MH.
5. **7th CPC Level 5/6 matrix**: authoritative URL.
6. **Karnataka fresh-entry BSc MLT**: which authority runs it (KEA UGCET / KNRUHS-equivalent / college-level)? Only lateral entry was verifiable.
7. **AP counselling**: canonical URL for Dr. NTR UHS paramedical (to split `counselling_telangana_ap`).
8. **MP paramedical notices**: the exact section path on `paramedical.mponline.gov.in`, and whether a session cookie is required.
9. **Vidya Lakshmi reachability**: confirm from the deployment egress; if it is only blocked from my test network, re-enable.
10. **State council websites**: for each of the ~30 NCAHP notifications, confirm a statutory site exists. Only Kerala (`kesahc.kerala.gov.in`) was verifiable. Add each to config **with the NCAHP notification URL as its source**, never from a search result alone.
11. **Scanned-vs-digital audit**: for the counselling seat matrices and state health-university college lists, the text layer was not tested. WS-D needs a per-family T0 hit-rate measurement before the T2 budget is set.

---

## 10. Change log

| Date | Change |
|---|---|
| 2026-11-16 | Initial spec. Based on live fetches documented in §1. No config or code touched. |
