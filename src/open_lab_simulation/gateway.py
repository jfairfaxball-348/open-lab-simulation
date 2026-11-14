"""Local adapter routing and immutable, experiment-selected provider catalogs."""

from __future__ import annotations

from . import ollama
from .contracts import canonical_json, sha256_text
from .model import DEFAULT_CONTEXT_TOKENS, DEFAULT_OUTPUT_TOKENS, is_loopback


def catalog(config, world):
    selected = config.get("extensions", {}).get("model_catalog")
    if selected is None:
        return world.providers, None
    if not isinstance(selected, dict) or set(selected) != {"document", "sha256"}:
        raise ValueError("model_catalog requires document and sha256")
    document = selected["document"]
    if sha256_text(canonical_json(document)) != selected["sha256"]:
        raise ValueError("model catalog digest differs from its pin")
    problems = world.contracts.validate(document, "model-provider")
    if problems:
        raise ValueError(f"invalid model catalog: {problems[0]}")
    ids = [m["id"] for m in document["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model ids in catalog")
    return {document["provider"]["id"]: document}, selected


def identify(provider, ref, *, timeout=30):
    api = provider["api"]
    if (provider["locality"] != "local" or not is_loopback(api.get("base_url", ""))
            or api["protocol"] != "openai_chat_completions"):
        raise ValueError("model server must be on this machine using loopback chat-completions")
    if provider["kind"] == "ollama":
        return ollama.identify(ollama.native_base(api["base_url"]), ref, timeout=timeout)
    if provider["kind"] not in {"openai_compatible", "llama_cpp"}:
        raise ValueError("unsupported local model adapter")
    listing = ollama._request(api["base_url"].rstrip("/"), "/models", timeout=timeout)
    entries = listing.get("data") if isinstance(listing, dict) else None
    if not isinstance(entries, list) or not any(
        isinstance(m, dict) and m.get("id") == ref for m in entries
    ):
        raise ValueError("model is not present on the local chat server")
    # A model name is not a weight digest; this API does not attest weight identity.
    return {"provider_ref": ref, "digest": None, "identity_limit": "model id only"}


def provenance(provider, model, sampling, identity, selected):
    if model["status"] != "available":
        raise ValueError("the model must be approved as available")
    if model.get("digest") and model["digest"] != identity.get("digest"):
        raise ValueError("model digest differs from the approved manifest")
    return {
        "provider_kind": provider["kind"],
        "adapter": {"id": "local-chat-completions", "version": "1.0.0"},
        "catalog": selected,
        "model_manifest": model,
        "provider_manifest": provider,
        "settings": {
            "context_guard_tokens": model.get("context_window_tokens", DEFAULT_CONTEXT_TOKENS),
            "effective_context_tokens": None,
            "context_limit": "client guard only; server context is not set or attested",
            "sampling_requested": sampling,
            "sampling_effective": None,
            "output_tokens_requested": sampling.get("max_output_tokens", DEFAULT_OUTPUT_TOKENS),
            "offload": None,
            "repair_policy": "syntactic-tool-repair-v1",
            "request_attempts_max": 2,
        },
    }
