"use client";

import { useEffect, useState } from "react";

import { parseStoredProfile, PROFILE_STORAGE_KEY, type Profile } from "./types";

/**
 * Read the profile from this browser's localStorage.
 *
 * `null` means "not read yet"; `{}` means "read, and the user answered
 * nothing". The value never leaves the browser except as a POST body to the
 * local API, and it is never written to a URL, cookie or log.
 */
export function useStoredProfile(): Profile | null {
  const [profile, setProfile] = useState<Profile | null>(null);

  useEffect(() => {
    function read() {
      setProfile(parseStoredProfile(window.localStorage.getItem(PROFILE_STORAGE_KEY)));
    }
    read();
    window.addEventListener("storage", read);
    return () => window.removeEventListener("storage", read);
  }, []);

  return profile;
}
