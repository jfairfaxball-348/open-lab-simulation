"""``ols`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, ollama
from .archive import verify_archive
from .paths import data_root
from .supervisor import LaunchError, run_experiment


def _cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.data_root) if args.data_root else data_root()
    try:
        result = run_experiment(Path(args.experiment), root)
    except LaunchError as error:
        print(f"Run refused; nothing was created. {error}", file=sys.stderr)
        return 2
    print(f"Run {result.run_id} ended: {result.state} ({result.reason})")
    print(f"  events:    {result.events}")
    print(f"  archive:   {result.archive}")
    print(f"  workspace: {result.workspace}")
    for note in result.notes:
        print(f"  note:      {note}")
    return 0 if result.state in ("completed", "terminated") else 1


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
        report = {"identity": ollama.identify(base, args.provider_ref),
                  "measured": ollama.measure(base, args.provider_ref)}
    except ollama.OllamaError as error:
        print(f"Measurement failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ols", description="Open Laboratory Simulation.")
    parser.add_argument("--version", action="version", version=f"ols {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Launch one run of an experiment and supervise it.")
    run.add_argument("experiment", help="Experiment configuration file (.json).")
    run.add_argument("--data-root", help="Data root. Default: $OPEN_LAB_DATA or ~/open-lab-data.")
    run.set_defaults(func=_cmd_run)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
