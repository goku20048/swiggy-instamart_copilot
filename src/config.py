"""Shared paths, constants and small I/O helpers used by every pipeline stage."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
POLICIES = DATA / "policies"
DB_PATH = DATA / "instamart.db"
OUTPUTS = ROOT / "outputs"
METRICS = OUTPUTS / "metrics"
STATE = ROOT / "state"
SQL_FILE = ROOT / "sql" / "queries.sql"

RANDOM_SEED = 42
SNAPSHOT = "2025-10-01 00:00:00"
LAPSE_CUTOFF = "2025-09-01"
LAPSE_WINDOW_END = "2025-10-01"
FINAL_K = 4


def setup_logging(level: int = logging.INFO) -> None:
    """Configure one consistent log format for every entry point."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)


def save_metrics(name: str, payload: dict[str, Any]) -> Path:
    """Write the computed figures of one stage to ``outputs/metrics/<name>.json``."""
    METRICS.mkdir(parents=True, exist_ok=True)
    path = METRICS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
