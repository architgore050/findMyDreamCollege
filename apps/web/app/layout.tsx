import type { Metadata } from "next";

import { Nav } from "@/components/nav";

import "./globals.css";

export const metadata: Metadata = {
  title: "findMyDreamCollege — sourced MLT programme evidence",
  description:
    "Inventory, evidence and shortlist for bachelor's-level Medical Laboratory Technology programmes in India. Every number resolves to a source.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-stone-50 text-stone-900 antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-stone-900 focus:px-3 focus:py-2 focus:text-sm focus:text-white"
        >
          Skip to content
        </a>
        <Nav />
        <main id="main" className="mx-auto w-full max-w-7xl px-4 pb-16 pt-6 sm:px-6">
          {children}
        </main>
        <footer className="border-t border-stone-200 bg-white">
          <div className="mx-auto flex w-full max-w-7xl flex-col gap-1 px-4 py-5 text-xs text-stone-600 sm:px-6">
            <p>
              Evidence-first inventory. Numbers are computed from extracted claims; prose is not a
              ranking signal. &ldquo;Unknown&rdquo; is reported as unknown, never as zero or a
              midpoint.
            </p>
            <p>
              v1 covers India only. Profile data stays in this browser — no accounts, no server
              storage.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
