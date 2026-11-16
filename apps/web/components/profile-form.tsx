"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { emptyDraft, fromProfile, toProfile, type ProfileDraft } from "@/lib/profile-draft";
import { parseStoredProfile, PROFILE_STORAGE_KEY, profileSchema } from "@/lib/types";

type SelectOption = { value: string; label: string };

function FieldFrame({
  htmlFor,
  label,
  hint,
  children,
}: {
  htmlFor: string;
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor={htmlFor} className="block text-xs font-semibold text-stone-800">
        {label}
      </label>
      {children}
      {hint ? (
        <p id={`${htmlFor}-hint`} className="text-[11px] text-stone-500">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

function MaybeText({
  id,
  label,
  value,
  onChange,
  placeholder,
  hint,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  hint?: string;
}) {
  const unknown = value === "unknown";
  return (
    <FieldFrame htmlFor={id} label={label} hint={hint}>
      <div className="flex flex-wrap items-center gap-2">
        <input
          id={id}
          type="text"
          className="min-w-40 flex-1"
          value={unknown ? "" : value}
          placeholder={placeholder ?? "not set"}
          disabled={unknown}
          aria-describedby={hint ? `${id}-hint` : undefined}
          onChange={(event) => onChange(event.target.value)}
        />
        <label className="flex shrink-0 items-center gap-1 text-[11px] text-stone-600">
          <input
            type="checkbox"
            id={`${id}-unknown`}
            checked={unknown}
            onChange={(event) => onChange(event.target.checked ? "unknown" : "")}
          />
          prefer not to say
        </label>
      </div>
    </FieldFrame>
  );
}

function MaybeNumber({
  id,
  label,
  value,
  onChange,
  hint,
  inputMode = "decimal",
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  hint?: string;
  inputMode?: "decimal" | "numeric";
}) {
  const unknown = value === "unknown";
  return (
    <FieldFrame htmlFor={id} label={label} hint={hint}>
      <div className="flex flex-wrap items-center gap-2">
        <input
          id={id}
          type="text"
          inputMode={inputMode}
          className="num w-32"
          value={unknown ? "" : value}
          placeholder="not set"
          disabled={unknown}
          aria-describedby={hint ? `${id}-hint` : undefined}
          onChange={(event) => onChange(event.target.value)}
        />
        <label className="flex shrink-0 items-center gap-1 text-[11px] text-stone-600">
          <input
            type="checkbox"
            id={`${id}-unknown`}
            checked={unknown}
            onChange={(event) => onChange(event.target.checked ? "unknown" : "")}
          />
          prefer not to say
        </label>
      </div>
    </FieldFrame>
  );
}

function MaybeSelect({
  id,
  label,
  value,
  onChange,
  options,
  hint,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: readonly SelectOption[];
  hint?: string;
}) {
  return (
    <FieldFrame htmlFor={id} label={label} hint={hint}>
      <select
        id={id}
        value={value}
        className="w-full max-w-64"
        aria-describedby={hint ? `${id}-hint` : undefined}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">not set</option>
        <option value="unknown">prefer not to say / unknown</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </FieldFrame>
  );
}

function CheckGroup({
  legend,
  hint,
  name,
  values,
  selected,
  onChange,
}: {
  legend: string;
  hint?: string;
  name: string;
  values: readonly SelectOption[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  function toggle(value: string, checked: boolean) {
    onChange(checked ? [...selected, value] : selected.filter((item) => item !== value));
  }
  return (
    <fieldset className="space-y-1">
      <legend className="text-xs font-semibold text-stone-800">{legend}</legend>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {values.map((option) => (
          <label key={option.value} className="flex items-center gap-1.5 text-xs text-stone-700">
            <input
              type="checkbox"
              name={name}
              value={option.value}
              checked={selected.includes(option.value)}
              onChange={(event) => toggle(option.value, event.target.checked)}
            />
            {option.label}
          </label>
        ))}
      </div>
      {hint ? <p className="text-[11px] text-stone-500">{hint}</p> : null}
    </fieldset>
  );
}

const credentialOptions = [
  { value: "degree", label: "degree (BSc MLT / MLS)" },
  { value: "diploma", label: "diploma (DMLT)" },
  { value: "lateral_entry", label: "lateral entry" },
  { value: "certificate", label: "certificate" },
] as const;

const testStatusOptions = [
  { value: "taken", label: "taken" },
  { value: "planned", label: "planned" },
  { value: "not_taken", label: "not taken" },
] as const;

const yesNoUnknown = [
  { value: "yes", label: "yes" },
  { value: "no", label: "no" },
] as const;

export function ProfileForm() {
  const [draft, setDraft] = useState<ProfileDraft>(emptyDraft);
  const [loaded, setLoaded] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [status, setStatus] = useState("");

  useEffect(() => {
    const stored = parseStoredProfile(window.localStorage.getItem(PROFILE_STORAGE_KEY));
    setDraft(fromProfile(stored));
    setLoaded(true);
  }, []);

  function patch<K extends keyof ProfileDraft>(key: K, value: ProfileDraft[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
    setStatus("");
  }

  function save() {
    const candidate = toProfile(draft);
    const result = profileSchema.safeParse(candidate);
    if (!result.success) {
      setErrors(
        result.error.issues.map(
          (issue) => `${issue.path.join(".") || "profile"}: ${issue.message}`,
        ),
      );
      setStatus("Profile not saved. Fix the listed fields and save again.");
      return;
    }
    setErrors([]);
    window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(result.data));
    const now = new Date();
    setSavedAt(now.toISOString());
    setStatus(
      Object.keys(result.data).length === 0
        ? "Saved an empty profile. Every field remains unknown until you answer it."
        : "Profile saved in this browser.",
    );
  }

  function clear() {
    window.localStorage.removeItem(PROFILE_STORAGE_KEY);
    setDraft(emptyDraft());
    setErrors([]);
    setSavedAt(null);
    setStatus("Saved profile deleted from this browser.");
  }

  const answered = Object.keys(toProfile(draft)).length;

  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault();
        save();
      }}
      noValidate
    >
      <div className="panel">
        <div className="panel-head">
          <h2 className="text-sm font-bold">Proceed from what you actually know</h2>
          <span className="num text-[11px] text-stone-600">
            {answered} field{answered === 1 ? "" : "s"} with an answer
          </span>
        </div>
        <p className="px-3 py-2 text-xs text-stone-700">
          Nothing is pre-filled. Every field accepts “prefer not to say”, which is scored as unknown
          (an interval), never as zero and never as an average. Unanswered stays unset.
        </p>
      </div>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Identity and target</legend>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <MaybeText
            id="nationality"
            label="Nationality (ISO-2)"
            value={draft.nationality}
            placeholder="e.g. IN"
            hint="Used for eligibility only. v1 covers India, so a non-IN nationality may leave eligibility unknown."
            onChange={(value) => patch("nationality", value)}
          />
          <MaybeNumber
            id="target_intake_year"
            label="Target intake year"
            value={draft.target_intake_year}
            inputMode="numeric"
            hint="No default year. A programme without this intake cycle fails the intake filter."
            onChange={(value) => patch("target_intake_year", value)}
          />
          <MaybeText
            id="target_field"
            label="Target field"
            value={draft.target_field}
            placeholder="e.g. mlt"
            hint="v1 only inventories MLT/MLS. Other fields are out of scope, not lower-ranked."
            onChange={(value) => patch("target_field", value)}
          />
        </div>
        <div className="mt-3">
          <CheckGroup
            legend="Credential levels you will consider"
            name="credential_levels"
            values={credentialOptions}
            selected={draft.credential_levels}
            onChange={(value) => patch("credential_levels", value)}
            hint="Unselected means no preference is applied yet; it does not silently imply degree-only."
          />
        </div>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Marks</legend>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <FieldFrame htmlFor="marks_board" label="Board">
            <input
              id="marks_board"
              type="text"
              value={draft.marks_board}
              placeholder="not set"
              onChange={(event) => patch("marks_board", event.target.value)}
            />
          </FieldFrame>
          <MaybeNumber
            id="marks_class12"
            label="Class 12 aggregate (%)"
            value={draft.marks_class12}
            onChange={(value) => patch("marks_class12", value)}
          />
          <MaybeNumber
            id="marks_pcb"
            label="PCB aggregate (%)"
            value={draft.marks_pcb}
            onChange={(value) => patch("marks_pcb", value)}
          />
        </div>
        <p className="mt-2 text-[11px] text-stone-500">
          Percentages are compared with published cutoffs only. Missing cutoffs produce “unknown”,
          not an assumed pass.
        </p>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">English test</legend>
        <div className="grid gap-3 sm:grid-cols-3">
          <MaybeSelect
            id="english_type"
            label="Test"
            value={draft.english_type}
            options={[
              { value: "IELTS", label: "IELTS" },
              { value: "TOEFL", label: "TOEFL" },
              { value: "PTE", label: "PTE" },
              { value: "Duolingo", label: "Duolingo" },
              { value: "other", label: "other" },
            ]}
            onChange={(value) => patch("english_type", value)}
          />
          <MaybeNumber
            id="english_score"
            label="Score"
            value={draft.english_score}
            onChange={(value) => patch("english_score", value)}
          />
          <MaybeSelect
            id="english_status"
            label="Status"
            value={draft.english_status}
            options={testStatusOptions}
            onChange={(value) => patch("english_status", value)}
          />
        </div>
        <p className="mt-2 text-[11px] text-stone-500">
          A test not taken is not a low score. It is a distinct status.
        </p>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Entrance exams</legend>
        {draft.entrance_exams.length === 0 ? (
          <p className="text-xs text-stone-600">
            No exams listed. Programmes that require an exam will report the travel/registration
            filter as unknown, not pass.
          </p>
        ) : (
          <div className="space-y-2">
            {draft.entrance_exams.map((exam, index) => (
              <div key={index} className="grid items-end gap-2 sm:grid-cols-[1fr_8rem_10rem_auto]">
                <FieldFrame htmlFor={`exam-${index}`} label={`Exam ${index + 1}`}>
                  <input
                    id={`exam-${index}`}
                    type="text"
                    value={exam.exam}
                    placeholder="e.g. MHT_CET"
                    onChange={(event) => {
                      const next = draft.entrance_exams.map((row, rowIndex) =>
                        rowIndex === index ? { ...row, exam: event.target.value } : row,
                      );
                      patch("entrance_exams", next);
                    }}
                  />
                </FieldFrame>
                <FieldFrame htmlFor={`exam-score-${index}`} label="Score">
                  <input
                    id={`exam-score-${index}`}
                    type="text"
                    inputMode="decimal"
                    className="num w-full"
                    value={exam.score === "unknown" ? "" : exam.score}
                    placeholder="not set"
                    onChange={(event) => {
                      const next = draft.entrance_exams.map((row, rowIndex) =>
                        rowIndex === index ? { ...row, score: event.target.value } : row,
                      );
                      patch("entrance_exams", next);
                    }}
                  />
                </FieldFrame>
                <FieldFrame htmlFor={`exam-status-${index}`} label="Status">
                  <select
                    id={`exam-status-${index}`}
                    value={exam.status}
                    className="w-full"
                    onChange={(event) => {
                      const next = draft.entrance_exams.map((row, rowIndex) =>
                        rowIndex === index ? { ...row, status: event.target.value } : row,
                      );
                      patch("entrance_exams", next);
                    }}
                  >
                    <option value="">not set</option>
                    <option value="unknown">prefer not to say</option>
                    <option value="taken">taken</option>
                    <option value="planned">planned</option>
                    <option value="not_taken">not taken</option>
                  </select>
                </FieldFrame>
                <button
                  type="button"
                  className="btn mb-0.5"
                  onClick={() =>
                    patch(
                      "entrance_exams",
                      draft.entrance_exams.filter((_, rowIndex) => rowIndex !== index),
                    )
                  }
                >
                  Remove<span className="sr-only"> exam {index + 1}</span>
                </button>
              </div>
            ))}
          </div>
        )}
        <button
          type="button"
          className="btn mt-2"
          onClick={() =>
            patch("entrance_exams", [...draft.entrance_exams, { exam: "", score: "", status: "" }])
          }
        >
          Add entrance exam
        </button>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Budget</legend>
        <div className="grid gap-3 sm:grid-cols-3">
          <FieldFrame htmlFor="budget_currency" label="Currency">
            <input
              id="budget_currency"
              type="text"
              value={draft.budget_currency}
              placeholder="not set"
              maxLength={3}
              className="w-24 uppercase"
              onChange={(event) => patch("budget_currency", event.target.value)}
            />
          </FieldFrame>
          <MaybeNumber
            id="budget_annual"
            label="Annual ceiling"
            value={draft.budget_annual}
            onChange={(value) => patch("budget_annual", value)}
            hint="No default amount exists."
          />
          <MaybeNumber
            id="budget_total"
            label="Total ceiling"
            value={draft.budget_total}
            onChange={(value) => patch("budget_total", value)}
          />
        </div>
        <p className="mt-2 text-[11px] text-stone-500">
          Budget is a hard cutoff only when the programme&apos;s TCOA lower bound exceeds the
          ceiling. Unknown cost never excludes a programme.
        </p>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Financing</legend>
        <div className="grid gap-3 sm:grid-cols-3">
          <MaybeSelect
            id="loan_required"
            label="Loan required"
            value={draft.loan_required}
            options={yesNoUnknown}
            onChange={(value) => patch("loan_required", value)}
          />
          <MaybeSelect
            id="family_income_band"
            label="Family income band"
            value={draft.family_income_band}
            options={[
              { value: "below_4.5L", label: "below ₹4.5L" },
              { value: "4.5-8L", label: "₹4.5–8L" },
              { value: "8-15L", label: "₹8–15L" },
              { value: "15L_plus", label: "₹15L and above" },
            ]}
            onChange={(value) => patch("family_income_band", value)}
          />
          <MaybeSelect
            id="collateral_available"
            label="Collateral available"
            value={draft.collateral_available}
            options={yesNoUnknown}
            onChange={(value) => patch("collateral_available", value)}
          />
        </div>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Living, travel and region</legend>
        <div className="space-y-3">
          <CheckGroup
            legend="Living preference"
            name="living_preference"
            values={[
              { value: "hostel", label: "hostel" },
              { value: "independent", label: "independent / PG" },
            ]}
            selected={draft.living_preference}
            onChange={(value) => patch("living_preference", value)}
            hint="Cost scenarios are always shown side by side and never summed."
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <MaybeSelect
              id="travel_for_exams"
              label="Willing to travel for exams"
              value={draft.travel_for_exams}
              options={yesNoUnknown}
              onChange={(value) => patch("travel_for_exams", value)}
            />
            <FieldFrame
              htmlFor="regions"
              label="Regions / states"
              hint="Comma-separated. Leave blank to apply no region preference."
            >
              <input
                id="regions"
                type="text"
                value={draft.regions}
                placeholder="not set"
                onChange={(event) => patch("regions", event.target.value)}
              />
            </FieldFrame>
          </div>
        </div>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Qualitative self-ratings (1–5)</legend>
        <div className="grid gap-3 sm:grid-cols-3">
          <MaybeNumber
            id="extracurriculars"
            label="Extracurriculars"
            value={draft.extracurriculars}
            inputMode="numeric"
            onChange={(value) => patch("extracurriculars", value)}
          />
          <MaybeNumber
            id="leadership"
            label="Leadership"
            value={draft.leadership}
            inputMode="numeric"
            onChange={(value) => patch("leadership", value)}
          />
          <MaybeNumber
            id="clinical_exposure"
            label="Clinical exposure"
            value={draft.clinical_exposure}
            inputMode="numeric"
            onChange={(value) => patch("clinical_exposure", value)}
          />
        </div>
        <p className="mt-2 text-[11px] text-stone-500">
          Most Indian MLT admissions do not weigh these. They are recorded so the scoring engine can
          say “not used” explicitly instead of pretending they matter.
        </p>
      </fieldset>

      <fieldset className="panel px-3 py-3">
        <legend className="px-1 text-sm font-bold">Risk appetite</legend>
        <MaybeSelect
          id="risk_appetite"
          label="How much uncertainty is acceptable?"
          value={draft.risk_appetite}
          options={[
            { value: "must_be_safe", label: "must be safe" },
            { value: "estimated_ok", label: "estimated is fine" },
            { value: "speculative_ok", label: "speculative is acceptable" },
          ]}
          onChange={(value) => patch("risk_appetite", value)}
        />
      </fieldset>

      <div className="panel sticky bottom-0 z-10 border-stone-400 bg-stone-50/95 backdrop-blur">
        <div className="flex flex-wrap items-center gap-3 px-3 py-3">
          <button type="submit" className="btn btn-primary">
            Save profile in this browser
          </button>
          <Link href="/shortlist" className="btn">
            Go to shortlist
          </Link>
          <button type="button" className="btn" onClick={clear}>
            Delete saved profile
          </button>
          <span className="num text-[11px] text-stone-600">
            {loaded
              ? savedAt
                ? `stored ${savedAt.slice(11, 19)}Z`
                : "not yet stored"
              : "reading browser storage…"}
          </span>
        </div>
        {status ? (
          <p role="status" className="px-3 pb-2 text-xs text-stone-800">
            {status}
          </p>
        ) : null}
        {errors.length > 0 ? (
          <div role="alert" className="border-t border-red-300 bg-red-50 px-3 py-2">
            <p className="text-xs font-bold text-red-950">The profile was not saved:</p>
            <ul className="list-disc pl-5 text-xs text-red-950">
              {errors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </div>
        ) : null}
        <p className="border-t border-stone-200 px-3 py-2 text-[11px] text-stone-600">
          Stored only in this browser via localStorage. No account, no server copy, no analytics.
          Clearing browser data deletes it.
        </p>
      </div>
    </form>
  );
}
