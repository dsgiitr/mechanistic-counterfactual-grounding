"""Small shared helpers: seeding, JSON lines, configuration."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def derive_seed(*parts):
    """Deterministic 63-bit seed from arbitrary hashable parts."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(obj):
    """JSON-safe copy: NaN/inf -> None, tuples -> lists, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(clean(record), allow_nan=False) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def load_config(path=None):
    """Load configs/paper.yaml (or another YAML file) as a dict."""
    import yaml
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "paper.yaml"
    with path.open() as stream:
        return yaml.safe_load(stream)


def done_ids(path, key="example_id"):
    """IDs already written to a JSONL file, so interrupted runs can resume."""
    return {str(row[key]) for row in read_jsonl(path) if key in row}
