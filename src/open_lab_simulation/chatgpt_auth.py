"""Sign in with ChatGPT for plan usage: registration, tokens, refresh, sign-out (S014d).

Implements OpenAI's documented flow for open-source and locally hosted apps
(developers.openai.com/siwc/token-sharing-open-source): dynamic client
registration through the owner's own browser, PKCE, a loopback callback on
127.0.0.1, ID-token validation against OpenAI's published keys, and a check that
plan usage (``chatgpt.tokens.use.direct``) was actually granted.

Nothing here automates the browser or reads cookies; the owner signs in and
consents themself. Tokens are stored with the DPAPI credential store under the
provider id and are never printed, logged or archived. The non-secret
registration (issued client id, host id, account subject) is kept beside them so
that sign-out can retain it for a later sign-in, as the documentation requires.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from . import credentials
from .ownership import archive_lock

PROVIDER_ID = "openai-chatgpt"
APP_NAME = "The Open Laboratory"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
CALLBACK_PATH = "/auth/callback"
FIRST_PORT = 1455
REFRESH_MARGIN_S = 300
SIGN_IN_TIMEOUT_S = 300
HTTP_TIMEOUT_S = 30
DYNAMIC_CLIENT = "dynamic_agent_client"
# Codes after which a refresh token can never work again (errors-and-recovery).
UNUSABLE_REFRESH = {"invalid_grant", "invalid_refresh_token", "token_expired",
                    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"}
# DER prefix of a PKCS#1 v1.5 DigestInfo for SHA-256 (RFC 8017, section 9.2).
_SHA256_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


class SignInError(Exception):
    """Sign-in, refresh or sign-out failed. Messages never contain a token."""


# --- small helpers ---------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _registration_path() -> Path:
    return credentials.credentials_dir() / f"{PROVIDER_ID}.registration.json"


def load_registration() -> dict[str, Any]:
    path = _registration_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        raise SignInError("the ChatGPT registration record is unreadable") from None


def _save_registration(record: dict[str, Any]) -> None:
    path = _registration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def host_id() -> str:
    """This installation's persisted opaque host id, created before its first sign-in."""
    registration = load_registration()
    if not registration.get("ext_agent_host_id"):
        registration["ext_agent_host_id"] = f"urn:uuid:{uuid.uuid4()}"
        _save_registration(registration)
    return registration["ext_agent_host_id"]


def _post_form(url: str, fields: dict[str, str]) -> tuple[int, Any]:
    request = urllib.request.Request(
        url, data=urllib.parse.urlencode(fields).encode("ascii"), method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw.decode("utf-8")) if raw else None
    except (UnicodeDecodeError, ValueError):
        return status, None


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _error_code(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or "unknown")
        if isinstance(error, str):
            return error
    return "unknown"


# --- ID token validation ---------------------------------------------------------------


def _rs256_valid(signing_input: bytes, signature: bytes, jwk: dict[str, Any]) -> bool:
    n = int.from_bytes(_unb64url(jwk["n"]), "big")
    e = int.from_bytes(_unb64url(jwk["e"]), "big")
    size = (n.bit_length() + 7) // 8
    if len(signature) != size:
        return False
    recovered = pow(int.from_bytes(signature, "big"), e, n).to_bytes(size, "big")
    digest = _SHA256_INFO + hashlib.sha256(signing_input).digest()
    expected = b"\x00\x01" + b"\xff" * (size - len(digest) - 3) + b"\x00" + digest
    return hmac.compare_digest(recovered, expected)


def validate_id_token(id_token: str, client_id: str, nonce: str | None,
                      keys: list[dict[str, Any]], now: float | None = None) -> dict[str, Any]:
    """Signature (RS256, published keys), issuer, audience, expiry and nonce."""
    try:
        header_b64, payload_b64, signature_b64 = id_token.split(".")
        header = json.loads(_unb64url(header_b64))
        claims = json.loads(_unb64url(payload_b64))
        signature = _unb64url(signature_b64)
    except (ValueError, TypeError):
        raise SignInError("the ID token is malformed") from None
    if header.get("alg") != "RS256":
        raise SignInError("the ID token uses an unexpected signature algorithm")
    candidates = [k for k in keys if k.get("kty") == "RSA"
                  and (header.get("kid") is None or k.get("kid") == header.get("kid"))]
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    if not any(_rs256_valid(signing_input, signature, key) for key in candidates):
        raise SignInError("the ID token signature does not verify")
    now = time.time() if now is None else now
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if claims.get("iss") != ISSUER:
        raise SignInError("the ID token issuer is wrong")
    if client_id not in audiences:
        raise SignInError("the ID token audience is not this client")
    if not isinstance(claims.get("exp"), int | float) or claims["exp"] + 60 < now:
        raise SignInError("the ID token has expired")
    if nonce is not None and not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
        raise SignInError("the ID token nonce does not match this sign-in")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise SignInError("the ID token has no subject")
    return claims


# --- sign-in ---------------------------------------------------------------------------


def authorization_url(*, client_id: str, redirect_uri: str, state: str, nonce: str,
                      challenge: str, host: str, id_token_hint: str | None = None) -> str:
    fields = {"client_id": client_id, "ext_agent_host_id": host, "response_type": "code",
              "redirect_uri": redirect_uri, "scope": SCOPES, "resource": RESOURCE,
              "state": state, "nonce": nonce, "code_challenge_method": "S256",
              "code_challenge": challenge}
    if client_id == DYNAMIC_CLIENT:
        fields["agent_name_hint"] = APP_NAME
    elif id_token_hint:
        fields["id_token_hint"] = id_token_hint
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(fields, quote_via=urllib.parse.quote)


class _Callback:
    """A one-shot loopback listener for the authorization redirect."""

    def __init__(self, port: int):
        self.result: dict[str, str] | None = None
        self.arrived = threading.Event()
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != CALLBACK_PATH or listener.arrived.is_set():
                    self.send_response(404)
                    self.end_headers()
                    return
                listener.result = dict(urllib.parse.parse_qsl(parsed.query))
                listener.arrived.set()
                page = (b"<!doctype html><title>The Open Laboratory</title>"
                        b"<p>Sign-in received. You can close this window and return to the "
                        b"terminal.</p>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *args):  # the query carries a code; never print it
                pass

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def sign_in(open_browser, *, port: int | None = None,
            timeout_s: float = SIGN_IN_TIMEOUT_S) -> dict[str, Any]:
    """Run the owner's interactive sign-in. ``open_browser(url)`` shows the page."""
    host = host_id()
    registration = load_registration()
    client_id = registration.get("client_id") or DYNAMIC_CLIENT
    hint = None
    if client_id != DYNAMIC_CLIENT:
        try:
            hint = json.loads(credentials.load(PROVIDER_ID)).get("id_token")
        except (credentials.CredentialError, ValueError):
            hint = None
    # Registration must use the documented port; later sign-ins may use any port.
    wanted = port if port is not None else (FIRST_PORT if client_id == DYNAMIC_CLIENT else 0)
    try:
        listener = _Callback(wanted)
    except OSError:
        raise SignInError(f"loopback port {wanted} is in use; free it and retry") from None
    redirect_uri = f"http://127.0.0.1:{listener.port}{CALLBACK_PATH}"
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    try:
        open_browser(authorization_url(client_id=client_id, redirect_uri=redirect_uri,
                                       state=state, nonce=nonce, challenge=challenge,
                                       host=host, id_token_hint=hint))
        if not listener.arrived.wait(timeout_s):
            raise SignInError("no sign-in callback arrived in time")
        result = listener.result or {}
    finally:
        listener.close()
    if not hmac.compare_digest(result.get("state", ""), state):
        raise SignInError("the callback state does not match this sign-in")
    if result.get("error"):
        raise SignInError(f"sign-in was not completed ({result['error']}); plan usage "
                          "stays disabled")
    returned = result.get("client_id")
    if client_id == DYNAMIC_CLIENT:
        if not returned or returned == DYNAMIC_CLIENT:
            raise SignInError("registration is incomplete: no issued client id was returned")
        client_id = returned
    elif returned and returned != client_id:
        raise SignInError("the callback named a different client; registration kept")
    if not result.get("code"):
        raise SignInError("the callback carried no authorization code")
    status, tokens = _post_form(TOKEN_URL, {
        "grant_type": "authorization_code", "client_id": client_id, "code": result["code"],
        "code_verifier": verifier, "redirect_uri": redirect_uri, "resource": RESOURCE})
    if status != 200 or not isinstance(tokens, dict):
        raise SignInError(f"code exchange failed (HTTP {status}, {_error_code(tokens)})")
    for name in ("access_token", "refresh_token", "id_token"):
        if not isinstance(tokens.get(name), str) or not tokens[name]:
            raise SignInError(f"the token response has no {name}")
    keys = _get_json(JWKS_URL).get("keys", [])
    claims = validate_id_token(tokens["id_token"], client_id, nonce, keys)
    if registration.get("subject") and registration["subject"] != claims["sub"]:
        raise SignInError("this is a different ChatGPT account from the registered one; "
                          "nothing was replaced")
    scopes = sorted(str(tokens.get("scope", "")).split())
    _store_tokens(tokens, scopes, client_id)
    registration.update({"client_id": client_id, "subject": claims["sub"],
                         "ext_agent_host_id": host, "plan_usage_granted": PLAN_SCOPE in scopes,
                         "signed_in_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _save_registration(registration)
    return {"client_id": client_id, "plan_usage_granted": PLAN_SCOPE in scopes,
            "scopes": scopes}


def _store_tokens(tokens: dict[str, Any], scopes: list[str], client_id: str) -> None:
    lifetime = tokens.get("expires_in")
    lifetime = lifetime if isinstance(lifetime, int | float) else 3600
    credentials.store(PROVIDER_ID, json.dumps({
        "client_id": client_id, "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"], "id_token": tokens.get("id_token"),
        "scopes": scopes, "expires_at": time.time() + lifetime,
    }, separators=(",", ":")))


# --- tokens at inference time ----------------------------------------------------------


def access_token() -> str:
    """A current access token, refreshed under a lock when near expiry.

    Raises ``CredentialError`` when plan usage is not usable (terminal for a run)
    and ``OSError`` when the issuer could not be reached (temporary; tokens kept).
    """
    directory = credentials.credentials_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with archive_lock(directory, recover_stale=True):
        try:
            record = json.loads(credentials.load(PROVIDER_ID))
        except ValueError:
            raise credentials.CredentialError("the stored ChatGPT session is unreadable") \
                from None
        if PLAN_SCOPE not in record.get("scopes", []):
            raise credentials.CredentialError("ChatGPT plan usage was not granted at sign-in")
        if record.get("expires_at", 0) - time.time() > REFRESH_MARGIN_S:
            return record["access_token"]
        status, tokens = _post_form(TOKEN_URL, {
            "grant_type": "refresh_token", "client_id": record["client_id"],
            "refresh_token": record["refresh_token"], "resource": RESOURCE})
        if status == 200 and isinstance(tokens, dict) and tokens.get("access_token") \
                and tokens.get("refresh_token"):
            scopes = sorted(str(tokens.get("scope") or " ".join(record["scopes"])).split())
            _store_tokens({**tokens, "id_token": tokens.get("id_token") or record.get("id_token")},
                          scopes, record["client_id"])
            if PLAN_SCOPE not in scopes:
                raise credentials.CredentialError("refreshed tokens lack ChatGPT plan usage")
            return tokens["access_token"]
        code = _error_code(tokens)
        if code in UNUSABLE_REFRESH:
            credentials.delete(PROVIDER_ID)
            raise credentials.CredentialError(
                f"the ChatGPT session ended ({code}); sign in again")
        if code == "invalid_client":
            raise credentials.CredentialError("the registered client is invalid (invalid_client)")
        if status >= 500:
            raise OSError(f"refresh unavailable (HTTP {status})")
        raise credentials.CredentialError(f"token refresh failed (HTTP {status}, {code})")


def sign_out() -> dict[str, Any]:
    """Revoke the renewable session, then clear local tokens; keep the registration."""
    try:
        record = json.loads(credentials.load(PROVIDER_ID))
    except (credentials.CredentialError, ValueError):
        return {"tokens_cleared": credentials.delete(PROVIDER_ID), "revoked": None}
    revoked = False
    for attempt in range(3):
        try:
            endpoint = _get_json(DISCOVERY_URL)["revocation_endpoint"]
            status, _ = _post_form(endpoint, {"token": record["refresh_token"],
                                              "token_type_hint": "refresh_token",
                                              "client_id": record["client_id"]})
        except (OSError, KeyError, ValueError, TypeError):
            status = None
        if status == 200:
            revoked = True
            break
        if status is not None and status < 500:
            break
        time.sleep(2 ** attempt)
    return {"tokens_cleared": credentials.delete(PROVIDER_ID), "revoked": revoked}


def list_models(token: str, base: str = RESOURCE) -> list[dict[str, str]]:
    """Models this account may use (``visibility == "list"``), in server order."""
    payload = _get_json(base + "/models", {"Authorization": f"Bearer {token}"})
    return [{"slug": m.get("slug"), "display_name": m.get("display_name")}
            for m in payload.get("models", []) if m.get("visibility") == "list"]
