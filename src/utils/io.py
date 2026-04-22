import json
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)
