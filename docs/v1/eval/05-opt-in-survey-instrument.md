# 05 — Opt-in graduate outcome survey: instrument design (NOT implementation)

**Wave:** FMD-CLOUD-024 / v1 eval builder
**Status:** design deliverable only. Nothing here is built; no survey infrastructure exists; no data
has been collected. This is the "deals study" instrument from `02-findings-viable.md` (P-later) and
`01-eval-designs.md` §2.6, written down so that the acquisition path (doc 06) and the legal review
have something concrete to approve or reject.

## 1. Purpose and honest framing

The product currently has **no programme-level outcome ground truth for Indian B.Sc. MLT** (doc 00,
§B). An opt-in survey of graduates is the only realistic path to self-reported outcome labels that
belong to a specific programme and cohort. Three honest limits, stated up front:

1. **Self-report is a proxy.** First-job title and employment status as reported by the graduate are
   not administrative records. The instrument is designed to make lying pointless and checking easy,
   but it cannot replace registration/placement data where that exists.
2. **Opt-in selection bias is structural.** Graduates of strong programmes with LinkedIn presence and
   time to answer are over-represented. The design therefore reports **coverage and response
   composition** alongside every rate, and the eval harness consumes survey rates only for
   programmes with a minimum respondent count and a documented composition.
3. **Cold start.** The first usable wave is at earliest 12–18 months after launch (doc 06's
   acquisition timeline). Nothing in v1 depends on it.

## 2. Data minimisation (DPDP-aligned)

The Digital Personal Data Protection Act, 2023 (India) requires purpose limitation and data
minimisation for personal data. The instrument collects **two facts and nothing else**:

| Field | Why it is needed | Retention |
|---|---|---|
| `programme_id` (institution + programme, from the product's own inventory) | the unit every metric aggregates over | retained (anonymised aggregate) |
| `graduation_year` | cohort dimension; outcome rates only make sense per cohort | retained (anonymised aggregate) |

Explicitly **never collected**: name, phone, email, registration/council number, marks, employer
identity beyond sector, current salary as an exact figure (bands only, see §4), IP address stored
with responses, or any third-party identifier. The survey is anonymous by construction — there is no
account, no login, no way to link a response to a product user, because the product does not retain
user profiles (invariant 1). Consent language states the purpose (aggregate outcome statistics for
prospective students), the voluntariness, and the right to have a response deleted **before
aggregation** (a rolling cut-off; after aggregation the response is no longer identifiable and
deletion requests cannot be honoured — stated in the consent text).

## 3. Wording — the graduate is asked about their own programme

The respondent is a graduate describing the programme they completed. The wording is neutral,
first-person, and never suggests an answer or ranks institutions. English wording (regional-language
variants are listed in §6; the English text is the master):

> **About your programme** — This survey asks only about the programme you completed and the year you
> graduated. No name, contact details, or registration number is collected. Answers are used only as
> anonymous, aggregated statistics to help future students compare programmes.
>
> 1. Which institution and programme did you complete? *(pick from list; free-text "other" routes to
>    manual mapping, never auto-published)*
> 2. In which year did you graduate? *(single select)*
> 3. Within 6 months of graduating, were you working in a role related to medical laboratory
>    science? *(Yes, in a lab role / Yes, in a related healthcare role / No, working outside the
>    field / Not yet — still looking / Not yet — further study)*
> 4. Did you complete the registration or licensure step required for your role? *(Yes / No / Not
>    required for my role / Unsure)*
> 5. Was your programme's internship placement arranged by the institution? *(Yes / No / Partly /
>    There was no internship)*
> 6. How well did the programme prepare you for your first role? *(5-point scale, "not at all" to
>    "very well")*
> 7. Optional, in bands: approximate monthly take-home in your first role. *(<₹10k / ₹10–15k /
>    ₹15–25k / ₹25–40k / >₹40k / Prefer not to say)*
> 8. Anything you would tell a student choosing this programme today? *(free text, optional,
>    moderated before any display; no personal data)*

Design notes: Q3–Q4 separate **field employment** from **registration conversion** because the
pipeline's invariant 6 (internships are not employment) has a survey analogue — "got placed" and
"got hired" must not blur. Q5 checks the institution-side claim against the graduate-side experience;
disagreement between the two is itself a quality signal. Q7 uses bands only, because exact salaries
invite fabrication and create re-identification risk in small cohorts.

## 4. Aggregation and minimum-cell rules

- A programme-cohort cell publishes a rate only with **n ≥ 8 respondents** and after the composition
  check (§5). Below that, the cell reports "insufficient responses" — the same UNKNOWN discipline the
  harness uses everywhere else.
- Rates are reported with their **response composition** (graduation-year spread, role-mix) and never
  as a bare percentage.
- Free-text answers are moderated; anything containing personal data or unverifiable accusations is
  dropped, and the moderation rule is published.

## 5. Incentives and retention approach

- **No payment for answers.** Paid surveys attract professional respondents and bias the pool.
  Instead: the aggregate "how graduates of this programme fared" page is shown to respondents
  immediately after submission (value-for-value exchange), and respondents may opt into a once-a-year
  re-contact **by providing an email into a separate, consent-scoped store** that is never joined to
  responses. That store is the only place any identifier exists, and it can be deleted wholesale.
- **Retention:** raw responses roll up into the anonymised aggregate on a fixed schedule (90 days);
  the separate re-contact store lives until the holder opts out or the wave ends. The aggregate is
  the only long-lived artefact and it is the thing the eval harness may eventually consume.
- **Incentive options considered and rejected:** sweepstakes (gambling-adjacent, consent mess),
  payment per response (bias), institutional kickbacks (an institution must never be able to influence
  its own cell — a hard red line).

## 6. Open questions this design leaves to the legal/acquisition review

1. Whether a **privacy notice page** (not just in-survey consent text) is required before the first
   wave, and who the data fiduciary of record is.
2. Regional-language versions (Hindi + the state languages of the first wave institutions) — the
   wording above is the master; translations need back-translation checks.
3. Whether "programme" should be pinned to the AISHE code where the institution has one, to survive
   institution renames.
4. The re-contact email store's lawful basis and its separation guarantees in the actual stack.

**Nothing in this document authorises collection.** The instrument exists so the owners of questions
1–4 can approve a concrete thing instead of an idea.
