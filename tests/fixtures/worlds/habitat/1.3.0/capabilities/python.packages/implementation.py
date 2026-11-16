"""Inspect installed distributions without importing their executable modules."""

from importlib.metadata import PackageNotFoundError, version

WORKSHOP = ("numpy", "scipy", "sympy", "mpmath", "pandas", "matplotlib", "pillow")


def run(arguments):
    packages = []
    for name in sorted(set(arguments.get("names", WORKSHOP))):
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        packages.append({"name": name, "version": installed})
    return {"packages": packages}
