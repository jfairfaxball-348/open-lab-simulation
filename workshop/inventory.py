"""Preserve exact distribution identities and wheel licence files in the image."""

import importlib.metadata
import json
import shutil
from pathlib import Path

ROOT = Path("/opt/open-lab-workshop")
packages = []
for line in (ROOT / "requirements.lock").read_text().splitlines():
    if not line or line[0].isspace() or "==" not in line:
        continue
    name, pinned = line.split()[0].split("==")
    dist = importlib.metadata.distribution(name)
    if dist.version != pinned:
        raise ValueError(f"installed {name} does not match its lock")
    notices = []
    for file in dist.files or []:
        if any(word in str(file).lower() for word in ("license", "copying", "notice")):
            source = Path(dist.locate_file(file))
            if source.is_file():
                target = ROOT / "notices" / name / str(file)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                notices.append(target.relative_to(ROOT).as_posix())
    packages.append({"name": name, "version": dist.version, "notices": sorted(notices),
                     "license": dist.metadata.get("License-Expression")
                     or dist.metadata.get("License", "See bundled distribution metadata")})
(ROOT / "inventory.json").write_text(json.dumps(packages, indent=2) + "\n")
