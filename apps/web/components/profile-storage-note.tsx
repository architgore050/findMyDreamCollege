/**
 * One visible sentence, on every screen that reads the profile. No analytics,
 * no server-side storage: the profile exists in this browser only (brief §2).
 */
export function ProfileStorageNote() {
  return (
    <p className="rounded border border-stone-300 bg-white px-3 py-1.5 text-[11px] text-stone-700">
      <span aria-hidden="true" className="mr-1 font-semibold">
        ●
      </span>
      Your profile stays in this browser. It is sent only in the request body to your local API and
      is never stored, logged, put in a URL or shared with analytics.
    </p>
  );
}
