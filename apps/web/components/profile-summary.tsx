"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { parseStoredProfile, PROFILE_STORAGE_KEY, type Profile } from "@/lib/types";

function summarise(profile: Profile): string[] {
  const rows: string[] = [];
  if (profile.target_intake_year !== undefined)
    rows.push(`intake ${String(profile.target_intake_year)}`);
  if (profile.target_field !== undefined) rows.push(`field ${String(profile.target_field)}`);
  if (profile.regions && profile.regions.length > 0)
    rows.push(`regions ${profile.regions.join(", ")}`);
  if (profile.credential_levels && profile.credential_levels.length > 0)
    rows.push(`credentials ${profile.credential_levels.join(", ")}`);
  if (profile.budget?.annual_ceiling !== undefined)
    rows.push(`annual ceiling ${String(profile.budget.annual_ceiling)}`);
  if (profile.travel_for_exams !== undefined)
    rows.push(`travel ${String(profile.travel_for_exams)}`);
  return rows;
}

/**
 * The profile lives only in localStorage (brief §2). This component is the only
 * place a screen reads it, and it never sends it to the Next.js server.
 */
export function ProfileSummary() {
  const [profile, setProfile] = useState<Profile | null>(null);

  useEffect(() => {
    setProfile(parseStoredProfile(window.localStorage.getItem(PROFILE_STORAGE_KEY)));
  }, []);

  if (profile === null) {
    return <p className="text-xs text-stone-600">Reading profile from this browser…</p>;
  }

  const answered = Object.keys(profile);
  if (answered.length === 0) {
    return (
      <div className="rounded border border-dashed border-stone-400 bg-white px-3 py-2 text-xs text-stone-700">
        <strong>No profile saved in this browser.</strong> Ranking needs at least a target intake
        year and field.{" "}
        <Link className="link" href="/">
          Fill in the profile
        </Link>
        . Nothing has been assumed on your behalf.
      </div>
    );
  }

  const rows = summarise(profile);
  return (
    <div className="rounded border border-stone-300 bg-white px-3 py-2 text-xs text-stone-700">
      <p>
        <strong>{answered.length}</strong> profile section{answered.length === 1 ? "" : "s"} stored
        in this browser ·{" "}
        <Link className="link" href="/">
          edit profile
        </Link>
      </p>
      {rows.length > 0 ? (
        <p className="num mt-1 text-[11px] text-stone-600">{rows.join(" · ")}</p>
      ) : (
        <p className="mt-1 text-[11px] text-stone-600">
          Stored profile has no target details yet; filters will report unknown.
        </p>
      )}
    </div>
  );
}
