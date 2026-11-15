"""``ols`` command-line interface."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from . import __version__, chatgpt, chatgpt_auth, credentials, gemini, ollama, providers  # noqa: F401
from .archive import verify_archive
from .paths import data_root
from .supervisor import LaunchError, recover_interrupted, resume_experiment, run_experiment


def _cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.data_root) if args.data_root else data_root()
    try:
        result = run_experiment(
            Path(args.experiment), root, pause_after_steps=args.pause_after_steps
        )
    except LaunchError as error:
        print(f"Run refused; nothing was created. {error}", file=sys.stderr)
        return 2
    return _print_result(result)


def _cmd_resume(args: argparse.Namespace) -> int:
    root = Path(args.data_root) if args.data_root else data_root()
    try:
        result = resume_experiment(
            Path(args.archive), root, pause_after_steps=args.pause_after_steps
        )
    except LaunchError as error:
        print(f"Resume refused. {error}", file=sys.stderr)
        return 2
    return _print_result(result)


def _print_result(result) -> int:
    print(f"Run {result.run_id} ended: {result.state} ({result.reason})")
    print(f"  events:    {result.events}")
    print(f"  archive:   {result.archive}")
    print(f"  workspace: {result.workspace}")
    for note in result.notes:
        print(f"  note:      {note}")
    return 0 if result.state in ("completed", "terminated", "paused") else 1


def _cmd_recover(args: argparse.Namespace) -> int:
    results, refused = recover_interrupted(Path(args.data_root) if args.data_root else None)
    for result in results:
        _print_result(result)
    for run_id, reason in refused:
        print(f"Recovery refused for {run_id}: {reason}", file=sys.stderr)
    print(f"Startup recovery: {len(results)} resumed, {len(refused)} refused")
    return 1 if refused or any(r.state == "failed" for r in results) else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    problems = verify_archive(archive)
    for problem in problems:
        print(f"        {problem}")
    print("Run archive FAILED verification" if problems else "Run archive is intact")
    return 1 if problems else 0


def _cmd_measure(args: argparse.Namespace) -> int:
    base = ollama.native_base(args.base_url)
    try:
        report = {
            "identity": ollama.identify(base, args.provider_ref),
            "measured": ollama.measure(base, args.provider_ref),
        }
    except ollama.OllamaError as error:
        print(f"Measurement failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _cmd_provider(args: argparse.Namespace) -> int:
    """Owner-only provider administration. Credentials are typed, never passed as arguments."""
    root = Path(args.data_root) if args.data_root else data_root()
    if args.action != "status" and not args.provider:
        print("Provider command refused: name the provider", file=sys.stderr)
        return 2
    try:
        if args.action == "set-key":
            secret = getpass.getpass(f"Credential for {args.provider} (input is hidden): ").strip()
            credentials.store(args.provider, secret)
            print(f"Stored an encrypted credential for {args.provider} ({len(secret)} characters).")
        elif args.action in ("sign-in", "sign-out", "models"):
            return _chatgpt_action(args)
        elif args.action == "clear-key":
            removed = credentials.delete(args.provider)
            print("Credential deleted." if removed else "No credential was stored.")
        elif args.action == "configure":
            enabled = True if args.enable else False if args.disable else None
            profile = providers.write_profile(
                root, args.provider, enabled=enabled, per_run=args.per_run,
                per_day=args.per_day, total=args.total, min_interval_s=args.min_interval,
                max_attempts_per_step=args.attempts, request_timeout_s=args.timeout)
            print(json.dumps(profile, indent=2))
        else:
            ids = [args.provider] if args.provider else sorted(providers.ADAPTERS)
            for ident in ids:
                try:
                    profile = providers.load_profile(root, ident)
                except providers.ProviderError as error:
                    profile = {"note": str(error)}
                used = providers.Ledger(root, ident).usage("", None)
                print(json.dumps({"provider": ident, "credential_stored": credentials.exists(ident),
                                  "profile": profile,
                                  "requests_reserved": {"last_24h": used["per_day"],
                                                        "total": used["total"]}}, indent=2))
    except (credentials.CredentialError, providers.ProviderError) as error:
        print(f"Provider command refused: {error}", file=sys.stderr)
        return 2
    return 0


def _chatgpt_action(args: argparse.Namespace) -> int:
    """Sign in with ChatGPT: the owner signs in in their own browser; tokens never print."""
    if args.provider != chatgpt_auth.PROVIDER_ID:
        print(f"Provider command refused: {args.action} applies only to "
              f"{chatgpt_auth.PROVIDER_ID}", file=sys.stderr)
        return 2
    try:
        if args.action == "sign-in":
            import webbrowser

            def show(url: str) -> None:
                print("Continue with ChatGPT: a browser window is opening. If it does not, "
                      "open the sign-in page printed below.", flush=True)
                if not webbrowser.open(url):
                    print(url, flush=True)

            result = chatgpt_auth.sign_in(show)
            print(json.dumps(result, indent=2))
            if not result["plan_usage_granted"]:
                print("ChatGPT plan usage was not granted; the provider stays unusable.",
                      file=sys.stderr)
                return 2
            print("Set this app's usage limit at https://chatgpt.com/settings/usage")
        elif args.action == "sign-out":
            result = chatgpt_auth.sign_out()
            print(json.dumps(result, indent=2))
            if result["revoked"] is False:
                print("Remote revocation was not confirmed; disconnect the app in ChatGPT "
                      "settings.", file=sys.stderr)
        else:
            for model in chatgpt_auth.list_models(chatgpt_auth.access_token()):
                print(json.dumps(model))
    except chatgpt_auth.SignInError as error:
        print(f"Sign-in refused: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Provider command failed: {type(error).__name__}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ols", description="Open Laboratory Simulation.")
    parser.add_argument("--version", action="version", version=f"ols {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Launch one run of an experiment and supervise it.")
    run.add_argument("experiment", help="Experiment configuration file (.json).")
    run.add_argument("--data-root", help="Data root. Default: $OPEN_LAB_DATA or ~/open-lab-data.")
    run.add_argument(
        "--pause-after-steps",
        type=int,
        help="Operator pause at this completed-step count; needs checkpointing.",
    )
    run.set_defaults(func=_cmd_run)

    resume = commands.add_parser("resume", help="Resume a stopped run from a verified checkpoint.")
    resume.add_argument("archive", help="Run archive beneath this data root's runs directory.")
    resume.add_argument("--data-root", help="Data root containing this archive and its world.")
    resume.add_argument("--pause-after-steps", type=int)
    resume.set_defaults(func=_cmd_resume)

    recover = commands.add_parser("recover", help="Startup scan of opted-in interrupted runs.")
    recover.add_argument("--data-root")
    recover.set_defaults(func=_cmd_recover)

    verify = commands.add_parser("verify", help="Check a run archive is complete and untampered.")
    verify.add_argument("archive", help="Run archive: <data root>/runs/<experiment>/<run>.")
    verify.set_defaults(func=_cmd_verify)

    measure = commands.add_parser(
        "measure-model",
        help="Load a model on a local Ollama server once and report memory, load time and speed.",
    )
    measure.add_argument("base_url", help="The provider's base URL, e.g. http://127.0.0.1:11434/v1")
    measure.add_argument("provider_ref", help="The server's name for the model, e.g. llama3.2:3b")
    measure.set_defaults(func=_cmd_measure)

    provider = commands.add_parser(
        "provider", help="Owner administration of hosted providers (disabled by default).")
    provider.add_argument("action", choices=("status", "set-key", "clear-key", "configure",
                                                   "sign-in", "sign-out", "models"))
    provider.add_argument("provider", nargs="?", help="Provider id, e.g. google-gemini.")
    provider.add_argument("--data-root")
    toggle = provider.add_mutually_exclusive_group()
    toggle.add_argument("--enable", action="store_true")
    toggle.add_argument("--disable", action="store_true")
    for flag in ("--per-run", "--per-day", "--total", "--attempts"):
        provider.add_argument(flag, type=int, help="Approved request ceiling or attempts.")
    provider.add_argument("--min-interval", type=float, help="Seconds between requests.")
    provider.add_argument("--timeout", type=float, help="Seconds per provider request.")
    provider.set_defaults(func=_cmd_provider)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
