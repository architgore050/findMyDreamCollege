import Link from "next/link";

export default function NotFound() {
  return (
    <div className="space-y-3">
      <p className="kicker">404</p>
      <h1 className="text-2xl font-bold tracking-tight">No such page</h1>
      <p className="max-w-2xl text-sm text-stone-700">
        The route does not exist in this build. The v1 screens are profile, shortlist, programme
        drill-down, compare, exams, exclusions, review and sources.
      </p>
      <p className="text-sm">
        <Link className="link" href="/">
          Start at the profile
        </Link>
      </p>
    </div>
  );
}
