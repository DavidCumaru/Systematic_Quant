"""
artifact.py
===========
Save backtest/walk-forward results as auditable, versionable JSON artifacts.
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_artifacts"


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def save_backtest_artifact(
    tickers: list[str],
    period_start: str,
    period_end: str,
    metrics: dict,
    params_used: dict,
    seed: int = 42,
    data_db_path: Path | None = None,
    extra: dict | None = None,
) -> Path:
    """
    Save a backtest result as a timestamped JSON artifact.

    Returns the path to the saved file.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticker_slug = "_".join(t.replace(".SA", "") for t in tickers[:4])
    if len(tickers) > 4:
        ticker_slug += f"_+{len(tickers)-4}"
    filename = f"backtest_{ticker_slug}_{ts}.json"

    artifact = {
        "timestamp":     datetime.now().isoformat(),
        "git_commit":    _git_commit_hash(),
        "seed":          seed,
        "period_start":  period_start,
        "period_end":    period_end,
        "tickers":       tickers,
        "params_used":   params_used,
        "metrics":       {k: _safe_value(v) for k, v in metrics.items()},
    }

    if data_db_path:
        artifact["data_hash"] = _file_hash(data_db_path)

    if extra:
        artifact["extra"] = extra

    path = ARTIFACTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)

    logger.info("Backtest artifact saved -> %s", path)
    return path


def _safe_value(v):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6) if not np.isnan(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
