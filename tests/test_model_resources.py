"""Deterministic resource-probe checks; no model or inhabitant."""

from types import SimpleNamespace

import pytest

from open_lab_simulation import model_resources


def test_cpu_capacity_needs_no_gpu_probe(monkeypatch):
    monkeypatch.setattr(model_resources.sys, "platform", "linux")
    monkeypatch.setattr(model_resources.os, "sysconf", lambda key: {
        "SC_AVPHYS_PAGES": 1024, "SC_PAGE_SIZE": 4096}[key], raising=False)
    monkeypatch.setattr(model_resources.shutil, "which", lambda name: None)
    assert model_resources.capacity(False) == {"ram_available_mb": 4, "vram_available_mb": 0}
    with pytest.raises(ValueError, match="unavailable"):
        model_resources.capacity(True)


@pytest.mark.parametrize("output", ["4000", "nan", "-1", "4000\n2000"])
def test_gpu_capacity_validates_measurement_and_uses_fixed_command(monkeypatch, output):
    monkeypatch.setattr(model_resources.sys, "platform", "linux")
    monkeypatch.setattr(model_resources.os, "sysconf", lambda key: 1024, raising=False)
    monkeypatch.setattr(model_resources.shutil, "which", lambda name: "nvidia-smi")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(model_resources.subprocess, "run", run)
    if output == "4000":
        assert model_resources.capacity()["vram_available_mb"] == 4000
    else:
        with pytest.raises(ValueError):
            model_resources.capacity()
    assert calls[0][0] == ["nvidia-smi", "--query-gpu=memory.free",
                           "--format=csv,noheader,nounits"]
    assert calls[0][1]["timeout"] == 2 and "shell" not in calls[0][1]
