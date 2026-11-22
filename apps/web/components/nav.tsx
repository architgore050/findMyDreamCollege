"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Profile" },
  { href: "/shortlist", label: "Shortlist" },
  { href: "/compare", label: "Compare" },
  { href: "/exams", label: "Exams" },
  { href: "/exclusions", label: "Exclusions" },
  { href: "/review", label: "Review" },
  { href: "/sources", label: "Sources" },
  { href: "/methodology", label: "Methodology" },
] as const;

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="border-b border-stone-300 bg-white">
      <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5 sm:px-6">
        <Link href="/" className="flex items-baseline gap-2 hover:underline">
          <span className="text-sm font-bold tracking-tight text-stone-900">
            findMyDreamCollege
          </span>
          <span className="text-[11px] text-stone-500">MLT · India · v1</span>
        </Link>
        <nav aria-label="Main" className="flex flex-wrap gap-1">
          {links.map((link) => {
            const active = isActive(pathname, link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={`rounded px-2 py-1 text-sm ${
                  active
                    ? "bg-stone-900 font-semibold text-white"
                    : "text-stone-700 hover:bg-stone-100 hover:text-stone-900"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
