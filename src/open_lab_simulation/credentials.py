"""Host-local provider credentials, encrypted for the current Windows user (ADR-0016).

Credentials live under the user profile, outside every repository, data root,
archive, workspace and container. They are written only by the interactive
``ols provider set-key`` command and read only by the supervisor's provider
broker. Nothing here ever prints, logs or returns a credential except to that
broker. Windows DPAPI binds the ciphertext to this user on this machine; other
platforms are refused rather than given a weaker store.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path

_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
# Application entropy: ciphertext copied elsewhere cannot be decrypted by other
# DPAPI callers of the same user that do not know it.
_ENTROPY = b"open-lab-provider-credential-v1"
_UI_FORBIDDEN = 0x1


class CredentialError(Exception):
    """A credential cannot be stored or read. The message never contains it."""


def credentials_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise CredentialError("LOCALAPPDATA is not set; no protected credential location")
    return Path(base) / "OpenLab" / "credentials"


def _path(provider_id: str) -> Path:
    if not (isinstance(provider_id, str) and _SLUG.match(provider_id)):
        raise CredentialError("invalid provider id")
    return credentials_dir() / f"{provider_id}.dpapi"


if sys.platform == "win32":
    from ctypes import wintypes

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]

    def _blob(data: bytes) -> tuple[_Blob, ctypes.Array]:
        buffer = ctypes.create_string_buffer(data, len(data))
        return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer

    def _call(function, data: bytes) -> bytes:
        source, keep = _blob(data)
        entropy, keep_entropy = _blob(_ENTROPY)
        out = _Blob()
        if not function(ctypes.byref(source), None, ctypes.byref(entropy), None, None,
                        _UI_FORBIDDEN, ctypes.byref(out)):
            raise CredentialError("Windows data protection failed")
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            _kernel32.LocalFree(out.pbData)
            del keep, keep_entropy

    def _protect(data: bytes) -> bytes:
        return _call(_crypt32.CryptProtectData, data)

    def _unprotect(data: bytes) -> bytes:
        return _call(_crypt32.CryptUnprotectData, data)
else:  # pragma: no cover - exercised only on non-Windows hosts
    def _protect(data: bytes) -> bytes:
        raise CredentialError("protected credential storage is implemented for Windows only")

    _unprotect = _protect


def store(provider_id: str, secret: str) -> None:
    if not isinstance(secret, str) or not secret or any(c.isspace() for c in secret):
        raise CredentialError("the credential is empty or contains whitespace")
    path = _path(provider_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(_protect(secret.encode("utf-8")))
    os.replace(temporary, path)


def load(provider_id: str) -> str:
    path = _path(provider_id)
    try:
        return _unprotect(path.read_bytes()).decode("utf-8")
    except FileNotFoundError:
        raise CredentialError(f"no credential is stored for '{provider_id}'") from None
    except (OSError, UnicodeDecodeError):
        raise CredentialError(f"the stored credential for '{provider_id}' is unreadable") from None


def exists(provider_id: str) -> bool:
    return _path(provider_id).is_file()


def delete(provider_id: str) -> bool:
    path = _path(provider_id)
    if path.is_file():
        path.unlink()
        return True
    return False
