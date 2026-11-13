"""Child-process side of the ``python`` capability interface.

Run as ``python -I -B _capability_runner.py <module:function>`` with the
capability directory as the working directory. Standard library only; it is
executed by path and imports nothing from the Simulation.

Protocol: argument object as JSON on stdin; exit 0 with the result as JSON on
stdout; exit 1 with ``{"error": message}`` when the capability raises
``ValueError``; any other exception is a crash (traceback on stderr, exit 2).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    module_name, function_name = sys.argv[1].split(":")
    arguments = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    try:
        path = Path.cwd().joinpath(*module_name.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location("capability_entrypoint", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            payload, status = getattr(module, function_name)(arguments), 0
        except ValueError as error:
            payload, status = {"error": str(error)}, 1
        encoded = json.dumps(payload).encode("utf-8")
    except Exception:  # noqa: BLE001 - every crash must become exit status 2
        traceback.print_exc()
        return 2
    sys.stdout.buffer.write(encoded)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
