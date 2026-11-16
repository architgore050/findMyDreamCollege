"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export function CompareForm({ initialIds }: { initialIds: string }) {
  const router = useRouter();
  const [value, setValue] = useState(initialIds);
  const [error, setError] = useState("");

  function submit() {
    const ids = value
      .split(",")
      .map((id) => id.trim())
      .filter((id) => id.length > 0);
    if (ids.length === 0) {
      setError("Enter at least one programme id.");
      return;
    }
    if (!ids.every((id) => /^\d+$/.test(id))) {
      setError("Programme ids are numeric. Use the id shown on a programme page.");
      return;
    }
    setError("");
    router.push(`/compare?ids=${ids.join(",")}`);
  }

  return (
    <form
      className="panel px-3 py-3"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <label htmlFor="compare-ids" className="block text-xs font-semibold text-stone-800">
        Programme ids to compare
      </label>
      <p id="compare-ids-hint" className="text-[11px] text-stone-500">
        Comma-separated database ids. The Pareto matrix is computed by the API (WS-F); this form
        only selects the records.
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        <input
          id="compare-ids"
          type="text"
          className="num min-w-48 flex-1"
          value={value}
          placeholder="e.g. 12, 41, 57"
          aria-describedby="compare-ids-hint"
          onChange={(event) => setValue(event.target.value)}
        />
        <button type="submit" className="btn btn-primary">
          Compare
        </button>
      </div>
      {error ? (
        <p role="alert" className="mt-2 text-xs font-semibold text-red-900">
          {error}
        </p>
      ) : null}
    </form>
  );
}
