import type { Metadata } from "next";

import { ProfileForm } from "@/components/profile-form";

export const metadata: Metadata = {
  title: "Profile — findMyDreamCollege",
};

export default function ProfilePage() {
  return (
    <div className="space-y-4">
      <header className="space-y-2">
        <p className="kicker">Step 1 of 3 · runtime input</p>
        <h1 className="text-2xl font-bold tracking-tight">Your profile</h1>
        <p className="max-w-3xl text-sm text-stone-700">
          The shortlist is a function of this profile. There is no sample applicant and no default
          budget, marks, country or intake year — every value here comes from you or stays unknown.
        </p>
        <dl className="flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-stone-600">
          <div className="flex gap-1">
            <dt className="font-semibold">Storage</dt>
            <dd>browser localStorage only</dd>
          </div>
          <div className="flex gap-1">
            <dt className="font-semibold">Accounts</dt>
            <dd>none in v1</dd>
          </div>
          <div className="flex gap-1">
            <dt className="font-semibold">Server copy</dt>
            <dd>none</dd>
          </div>
        </dl>
      </header>
      <ProfileForm />
    </div>
  );
}
