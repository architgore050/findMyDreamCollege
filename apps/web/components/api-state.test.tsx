import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ApiFailurePanel } from "@/components/api-state";

describe("API degradation", () => {
  it("renders a 501 as a named not-yet-available state, not a spinner", () => {
    render(
      <ApiFailurePanel
        result={{
          state: "not_implemented",
          endpoint: "/api/rankings",
          message: "/api/rankings is not implemented (WS-F, see brief §11)",
          workstream: "WS-F",
          meta: null,
        }}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Not yet available — owned by WS-F" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/is not implemented \(WS-F/)).toBeInTheDocument();
    expect(screen.getByText("/api/rankings")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /source registry/i })).toHaveAttribute(
      "href",
      "/sources",
    );
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("renders a database outage with recovery steps", () => {
    render(
      <ApiFailurePanel
        result={{
          state: "database_unavailable",
          endpoint: "/api/programmes",
          message: "database unavailable: connection refused",
          meta: null,
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Database unavailable" })).toBeInTheDocument();
    expect(screen.getByText(/connection refused/)).toBeInTheDocument();
    expect(screen.getByText("make migrate")).toBeInTheDocument();
  });

  it("renders an unexpected failure with its code and status", () => {
    render(
      <ApiFailurePanel
        result={{
          state: "error",
          endpoint: "/api/programmes/999",
          code: "not_found",
          message: "programme 999 not found",
          status: 404,
          meta: null,
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Request failed" })).toBeInTheDocument();
    expect(screen.getByText(/HTTP 404 · not_found/)).toBeInTheDocument();
  });
});
