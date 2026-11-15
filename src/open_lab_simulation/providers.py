"""Hosted model providers: profiles, the reservation ledger and the broker (ADR-0016).

The agent runtime never talks to a hosted provider. It talks to this broker, a
loopback server inside the supervisor process, in the gateway's existing
chat-completions shape, authenticated with a per-run token. The broker alone
reads the provider credential, reserves budget in a crash-safe ledger before
every outgoing attempt (retries included), sends one request at a time, and
translates through a versioned adapter. Failures end the model step with a
coded error; nothing ever falls back to another model, provider or tier.

Provider use is disabled by default. A host-local profile, written only by the
``ols provider configure`` command on the owner's instruction, enables it and
carries the approved ceilings. The experiment cannot raise them.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from . import credentials
from .contracts import canonical_json, sha256_text
from .ownership import archive_lock

_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
LIMIT_NAMES = ("per_run", "per_day", "total")
DAY_S = 24 * 3600
MAX_BACKOFF_S = 60
PROFILE_DEFAULTS = {"max_attempts_per_step": 2, "min_interval_s": 0.0, "request_timeout_s": 180}


class ProviderError(Exception):
    """A provider cannot be used as configured. Launch must be refused."""


# --- adapters ------------------------------------------------------------------------


@dataclass
class Upstream:
    """One outgoing request, without its credential."""

    url: str
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Outcome:
    """What the adapter made of one upstream response."""

    kind: str  # "ok", "retry" or "terminal"
    code: str
    reply: dict[str, Any] | None = None  # chat-completions shaped, for "ok"
    usage: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    finish_reason: str | None = None
    retry_after_s: float | None = None
    state: dict[str, Any] = field(default_factory=dict)  # adapter memory for later turns


class Adapter(Protocol):
    id: str
    version: str
    host: str

    def request(self, chat: dict[str, Any], settings: dict[str, Any],
                state: dict[str, Any]) -> Upstream: ...

    def auth_headers(self, secret: str) -> dict[str, str]: ...

    def response(self, status: int, payload: Any, headers: dict[str, str],
                 state: dict[str, Any]) -> Outcome: ...

    def settings_support(self, sampling: dict[str, Any],
                         inference: dict[str, Any]) -> dict[str, str]: ...


ADAPTERS: dict[str, Adapter] = {}


def adapter_for(provider_id: str) -> Adapter:
    try:
        return ADAPTERS[provider_id]
    except KeyError:
        raise ProviderError(f"no hosted adapter is implemented for provider '{provider_id}'") \
            from None


# --- profiles ------------------------------------------------------------------------


def provider_dir(root: Path, provider_id: str) -> Path:
    if not (isinstance(provider_id, str) and _SLUG.match(provider_id)):
        raise ProviderError("invalid provider id")
    return root / "providers" / provider_id


def load_profile(root: Path, provider_id: str) -> dict[str, Any]:
    path = provider_dir(root, provider_id) / "profile.json"
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProviderError(f"provider '{provider_id}' is not configured (disabled)") from None
    except (OSError, ValueError):
        raise ProviderError(f"provider '{provider_id}' profile is unreadable") from None
    return validate_profile(profile, provider_id)


def validate_profile(profile: Any, provider_id: str) -> dict[str, Any]:
    if not isinstance(profile, dict) or profile.get("provider_id") != provider_id:
        raise ProviderError("provider profile does not belong to this provider")
    limits = profile.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(LIMIT_NAMES) or any(
        type(limits[name]) is not int or limits[name] < 0 for name in LIMIT_NAMES
    ):
        raise ProviderError("provider profile needs integer per_run, per_day and total limits")
    merged = {**PROFILE_DEFAULTS, **profile}
    if type(merged["max_attempts_per_step"]) is not int or not 1 <= merged[
            "max_attempts_per_step"] <= 5:
        raise ProviderError("max_attempts_per_step must be 1 to 5")
    for name in ("min_interval_s", "request_timeout_s"):
        if type(merged[name]) not in (int, float) or not 0 <= merged[name] <= 3600:
            raise ProviderError(f"{name} must be 0 to 3600 seconds")
    if type(merged.get("enabled")) is not bool:
        raise ProviderError("provider profile needs a boolean 'enabled'")
    return merged


def write_profile(root: Path, provider_id: str, **settings) -> dict[str, Any]:
    directory = provider_dir(root, provider_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "profile.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "provider_id": provider_id, "enabled": False,
        "limits": {name: 0 for name in LIMIT_NAMES}}
    for name in LIMIT_NAMES:
        if settings.get(name) is not None:
            current["limits"][name] = settings[name]
    for name in ("enabled", *PROFILE_DEFAULTS):
        if settings.get(name) is not None:
            current[name] = settings[name]
    current["approved_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    validate_profile(current, provider_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return current


# --- ledger --------------------------------------------------------------------------


class Ledger:
    """Append-only, fsynced provider accounting under the data root.

    A reservation is written and synced before its request is sent and is never
    released: an attempt whose outcome is unknown counts as spent.
    """

    def __init__(self, root: Path, provider_id: str):
        self.directory = provider_dir(root, provider_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "ledger.jsonl"

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        if raw and not raw.endswith(b"\n"):
            # A torn final write still reserved budget; count it rather than drop it.
            lines[-1] = json.dumps({"kind": "reserve", "torn": True, "time": time.time()})
        return [json.loads(line) for line in lines if line]

    def usage(self, run_id: str, campaign: str | None) -> dict[str, int]:
        now = time.time()
        reserves = [e for e in self.entries() if e["kind"] == "reserve"]
        return {
            "per_run": sum(e.get("run_id") == run_id for e in reserves),
            # A rolling 24 hours is never looser than any calendar day in any zone.
            "per_day": sum(now - e.get("time", now) < DAY_S for e in reserves),
            "total": len(reserves),
            "campaign": sum(campaign is not None and e.get("campaign") == campaign
                            for e in reserves),
        }

    def exhausted(self, limits: dict[str, int], run_id: str, campaign: str | None) -> str | None:
        used = self.usage(run_id, campaign)
        for name in LIMIT_NAMES:
            if used[name] >= limits[name]:
                return name
        return None

    def append(self, entry: dict[str, Any], mirror: Path | None = None) -> None:
        line = (canonical_json(entry) + "\n").encode("utf-8")
        for path in (self.path, mirror):
            if path is None:
                continue
            with path.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())


# --- broker --------------------------------------------------------------------------


@dataclass
class Plan:
    """Everything the broker needs for one run, frozen at launch."""

    provider_id: str
    model_ref: str
    sampling: dict[str, Any]
    inference: dict[str, Any]
    profile: dict[str, Any]
    run_id: str
    campaign: str | None
    archive: Path


def _error(status: int, code: str, message: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"code": code, "message": message}}


class Broker:
    """Loopback chat-completions endpoint for one run, backed by one hosted adapter."""

    def __init__(self, root: Path, plan: Plan, adapter: Adapter | None = None):
        self.plan = plan
        self.adapter = adapter or adapter_for(plan.provider_id)
        self.ledger = Ledger(root, plan.provider_id)
        self.mirror = plan.archive / "provider-ledger.jsonl"
        self.token = secrets.token_urlsafe(32)
        self.state: dict[str, Any] = {}
        self.exhausted: str | None = None
        self.terminal: str | None = None
        self._lock = threading.Lock()
        self._last_send = 0.0
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> Broker:
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied.encode(), f"Bearer {broker.token}".encode()):
                    status, body = _error(401, "laboratory_unauthorised", "invalid run token")
                elif self.path != "/v1/chat/completions":
                    status, body = _error(404, "laboratory_route", "unknown route")
                else:
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        chat = json.loads(self.rfile.read(length).decode("utf-8"))
                        status, body = broker.complete(chat)
                    except (ValueError, UnicodeDecodeError):
                        status, body = _error(400, "laboratory_bad_request", "invalid JSON")
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # requests are recorded in the ledger, not stderr
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def complete(self, chat: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """One model step: up to max_attempts_per_step reserved provider attempts."""
        with self._lock, archive_lock(self.ledger.directory, recover_stale=True):
            profile, plan = self.plan.profile, self.plan
            if self.terminal:
                return _error(409, "laboratory_provider_stopped", self.terminal)
            last = "no attempt"
            for attempt in range(1, profile["max_attempts_per_step"] + 1):
                limit = self.ledger.exhausted(profile["limits"], plan.run_id, plan.campaign)
                if limit:
                    self.exhausted = limit
                    return _error(429, "laboratory_budget_exhausted",
                                  f"approved provider request budget '{limit}' is exhausted")
                upstream = self.adapter.request(chat, {**plan.sampling, "model": plan.model_ref},
                                                {**self.state, "inference": plan.inference})
                wait = self._last_send + profile["min_interval_s"] - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                ident = secrets.token_hex(8)
                self.ledger.append({
                    "kind": "reserve", "id": ident, "time": time.time(), "run_id": plan.run_id,
                    "campaign": plan.campaign, "attempt": attempt,
                    "adapter": f"{self.adapter.id}@{self.adapter.version}",
                    "model": plan.model_ref,
                    "body_sha256": sha256_text(canonical_json(upstream.body)),
                }, self.mirror)
                with self.mirror.open("ab") as stream:
                    stream.write((canonical_json({"kind": "request", "id": ident,
                                                  "url": upstream.url, "body": upstream.body})
                                  + "\n").encode("utf-8"))
                self._last_send = time.monotonic()
                started = time.monotonic()
                outcome = self._send(upstream)
                self.ledger.append({
                    "kind": "outcome", "id": ident, "time": time.time(), "result": outcome.kind,
                    "code": outcome.code, "usage": outcome.usage,
                    "model_version": outcome.model_version, "finish_reason": outcome.finish_reason,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }, self.mirror)
                if outcome.kind == "ok":
                    self.state.update(outcome.state)
                    return 200, {**outcome.reply, "usage": outcome.usage}
                last = outcome.code
                if outcome.kind == "terminal":
                    self.terminal = f"provider refused: {outcome.code}"
                    return _error(424, "laboratory_provider_terminal", self.terminal)
                if attempt < profile["max_attempts_per_step"]:
                    time.sleep(min(outcome.retry_after_s or 2 ** attempt, MAX_BACKOFF_S))
            return _error(424, "laboratory_provider_unavailable",
                          f"no usable provider reply after {profile['max_attempts_per_step']} "
                          f"attempts ({last})")

    def _send(self, upstream: Upstream) -> Outcome:
        # OAuth adapters supply ``secret()`` to refresh tokens; key adapters use the store.
        load_secret = getattr(self.adapter, "secret", None) or (
            lambda: credentials.load(self.plan.provider_id))
        try:
            secret = load_secret()
        except credentials.CredentialError as error:
            return Outcome("terminal", f"credential_unavailable: {error}")
        except OSError as error:
            # A refresh that could not reach the issuer is temporary; tokens are kept.
            return Outcome("retry", f"credential_refresh_{type(error).__name__}")
        request = urllib.request.Request(
            upstream.url, data=json.dumps(upstream.body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", **upstream.headers,
                     **self.adapter.auth_headers(secret)},
        )
        del secret
        if urllib.parse.urlsplit(upstream.url).hostname != self.adapter.host:
            return Outcome("terminal", "egress_host_not_permitted")
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=self.plan.profile["request_timeout_s"]) as response:
                status, raw, headers = response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            status, raw, headers = error.code, error.read(), dict(error.headers or {})
        except (OSError, TimeoutError) as error:
            # Sent or not, the reservation stands: the outcome is unknown.
            return Outcome("retry", f"transport_{type(error).__name__}")
        decode = getattr(self.adapter, "decode", None)
        if decode is not None:  # e.g. a server-sent event stream
            payload = decode(status, raw, headers)
        else:
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else None
            except (UnicodeDecodeError, ValueError):
                payload = None
        return self.adapter.response(status, payload, headers, self.state)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirects are refused", headers, fp)


def launch_check(root: Path, provider_id: str, run_id: str | None = None,
                 campaign: str | None = None) -> dict[str, Any]:
    """Refuse a hosted launch unless enabled, budgeted, credentialed and implemented."""
    adapter = adapter_for(provider_id)
    profile = load_profile(root, provider_id)
    if not profile["enabled"]:
        raise ProviderError(f"provider '{provider_id}' is disabled")
    if not credentials.exists(provider_id):
        raise ProviderError(f"no credential is stored for provider '{provider_id}'")
    limit = Ledger(root, provider_id).exhausted(profile["limits"], run_id or "", campaign)
    if limit:
        raise ProviderError(f"provider '{provider_id}' budget '{limit}' is already exhausted")
    return {"profile": profile, "adapter": adapter}

