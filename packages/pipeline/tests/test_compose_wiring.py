"""Offline pins for the compose network wiring (wave v1-w1).

The api container was observed looping ``Temporary failure in name resolution``
for ``db`` while the db ran under a *different* compose project: compose DNS
resolves service names only among containers attached to the same network, and
the project-default network name varies with the project. The fix attaches
every service to one declared network with a fixed *physical* name
(``fmd-net``), so any invocation of this compose file converges on one segment.

These tests run without docker: they parse ``infra/docker-compose.yml`` and
``scripts/acceptance.sh`` and pin the wiring invariants. The live probe is
``scripts/acceptance.sh`` section 0 (``compose/smoke``); a real ``make up`` on
the operator machine is LQ-1 in docs/v1/LOCAL-QUEUE.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_PATH = REPO_ROOT / "infra" / "docker-compose.yml"
ACCEPTANCE_PATH = REPO_ROOT / "scripts" / "acceptance.sh"

# Every service declared in the file, regardless of profile. The profiles
# (discovery, ocr, ocr-paddle) are opt-in, but they still must not drift onto
# their own network segment.
ALL_SERVICES = {"db", "searxng", "ocr", "ocr-paddle", "api", "worker", "web"}


def load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_file_exists_and_parses() -> None:
    data = load_compose()
    assert isinstance(data, dict) and "services" in data


def test_project_name_is_pinned() -> None:
    # A pinned project name stops two checkouts/directories from silently
    # creating two stacks ("fmd_db" vs "findmydreamcollege_db").
    name = load_compose().get("name")
    assert isinstance(name, str) and name.strip(), (
        "infra/docker-compose.yml must pin a project `name:`"
    )


def test_declared_network_has_fixed_physical_name() -> None:
    networks = load_compose().get("networks") or {}
    fmd_net = networks.get("fmd-net")
    assert isinstance(fmd_net, dict), (
        "compose must declare a `fmd-net` network; project-default networks "
        "split when services are started under different compose projects"
    )
    assert fmd_net.get("name") == "fmd-net", (
        "the network's physical name must be fixed (`name: fmd-net`) so a "
        "service started under a different project still lands on the same segment"
    )


def test_every_service_attaches_fmd_net() -> None:
    services = load_compose()["services"]
    # Exact equality, not subset: a future 8th service must not silently come
    # up on its own project-default network segment.
    assert set(services) == ALL_SERVICES, (
        f"compose services drifted: {set(services) ^ ALL_SERVICES} — update "
        "ALL_SERVICES and attach every new service to fmd-net"
    )
    for svc in sorted(ALL_SERVICES):
        nets = services[svc].get("networks")
        assert nets is not None and "fmd-net" in nets, (
            f"service {svc} must attach networks: [fmd-net] — a service left on "
            "the project-default network reintroduces the DNS split"
        )


def test_no_service_uses_host_network_mode() -> None:
    # `build.network: host` (BuildKit-only) is allowed; a runtime
    # `network_mode: host` would bypass compose DNS entirely.
    services = load_compose()["services"]
    for svc, spec in services.items():
        assert not spec.get("network_mode"), (
            f"service {svc} sets network_mode: {spec['network_mode']}; runtime "
            "host networking bypasses the declared network and its DNS"
        )


def test_api_database_url_targets_the_db_service() -> None:
    url = load_compose()["services"]["api"]["environment"]["DATABASE_URL"]
    assert re.search(r"@db:5432/", url), (
        f"api DATABASE_URL must name the db *service* (got {url!r}); a literal "
        "host/ip breaks the declared-network wiring"
    )


def test_startup_dependency_chain_is_health_gated() -> None:
    services = load_compose()["services"]
    assert services["api"]["depends_on"]["db"]["condition"] == "service_healthy"
    assert services["worker"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["web"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_db_has_healthcheck_and_restart_policy() -> None:
    db = load_compose()["services"]["db"]
    assert db.get("healthcheck"), "db must carry a healthcheck (depends_on gates on it)"
    assert db.get("restart") == "unless-stopped", (
        "db must restart on crash — api/worker health-gate on it and a silent db "
        "exit otherwise wedges the whole stack"
    )


def test_web_publishes_a_port() -> None:
    ports = load_compose()["services"]["web"].get("ports") or []
    assert ports, "web must publish a host port (definition of done: reachable web)"
    assert any(str(p).endswith(":3000") for p in ports), f"unexpected web ports: {ports}"


def test_api_host_gateway_and_dns_wiring_is_preserved() -> None:
    api = load_compose()["services"]["api"]
    assert "host.docker.internal:host-gateway" in api.get("extra_hosts", []), (
        "api must keep the host-gateway alias (Linux has no host.docker.internal)"
    )
    assert api.get("dns"), (
        "api keeps an explicit dns override: on hosts whose resolv.conf carries "
        "the systemd-resolved stub (127.0.0.53), a container cannot reach it"
    )


def test_acceptance_script_registers_the_compose_smoke_check() -> None:
    script = ACCEPTANCE_PATH.read_text(encoding="utf-8")
    assert "compose/smoke" in script, (
        "scripts/acceptance.sh must register the compose/smoke check — its "
        "absence means the wiring fix is never probed live"
    )
    assert "SKIP" in script and "compose file valid; stack not running" in script, (
        "compose/smoke must SKIP (with a reason) when docker or the stack is absent"
    )
