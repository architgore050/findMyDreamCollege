# Exam registration windows — feature spec (v1 requirement, added 2026-11-18)

The tool already distinguishes *university* application windows. It must equally track the
**registration windows for every exam score an applicant needs** — because a programme can be
open while its required exam is closed, which makes the programme inaccessible regardless of
eligibility.

## Why this is a hard filter, not a score

`AGENTS.md` invariant 4: cutoffs stay binary. An exam whose registration has closed cannot be
entered by this cycle's applicant, no matter how good the programme is. It must **exclude**,
with the deciding claim shown — never appear as a mild penalty.

## Data model (already mostly present)

`exams` and `exam_windows` exist (`db/migrations/0001`). Each window carries:
`cycle_year`, `registration_opens`, `registration_closes`, `sitting_dates`, `source_claim_id`,
`observed_at`.

Required semantics:

1. **Two different dates, never conflated:** *registration close* (the cutoff that filters) vs
   *exam sitting date* (informational). A programme is excluded on registration close only.
2. **Per cycle, per exam, per state/scope.** State counselling exams are scope-bound; national
   exams are not. Multiple concurrent cycles must coexist (e.g. 2026 late round + 2027 main).
3. **Observability and staleness.** Every window is a claim with `quote`, `observed_at`, TTL,
   and source URL. Past TTL ⇒ `unknown` ⇒ UI shows "verify", never "open". This is the same
   rule used for programme deadlines and it must not regress.
4. **Timezone discipline.** Indian deadlines are IST (Asia/Kolkata). Store UTC + original text;
   a deadline printed as "up to 5:00 pm on 12/03/2027" must not shift a day.
5. **Extensions and re-notifications.** Later notifications supersede earlier ones
   (`superseded_by`), keeping both for audit. A reopened window must flip the filter back to
   `pass` automatically on the next run.
6. **No invention.** If a notification gives only a month ("March 2027"), record it as a
   window with `precision: month` and widen the filter to `unknown` inside that month rather
   than fabricating a day.

## Where the data comes from

Official sources only, already in the registry: state counselling authorities, NTA (NEET/CUET),
university-level test pages, and `exam_notification` documents discovered by the crawler.
Past-paper links remain a separate, link-only field (never mirrored).

## Product surface

- `/exams`: each exam with cycle, registration open/close (with `observed_at` and staleness),
  sitting dates, travel requirement, and the programmes that depend on it.
- Shortlist: a programme whose required exam is closed is excluded with reason
  `exam_registration_closed` and the deciding claim id.
- Also surface the **next actionable deadline** for the user's profile, sorted ascending —
  this is the most useful single line the product can show a family with a month to act.

## Acceptance (for the workstream)

- Windows extracted from a real notification PDF with quote + page, and stored per cycle.
- A closed window excludes; an open one passes; a stale one returns `unknown` and renders
  "verify" (never "open").
- Superseding notification reopens the window without manual edits.
- Month-precision windows produce `unknown`, not a fabricated date.
- `/exams` and the shortlist exclusion both cite the deciding claim.
