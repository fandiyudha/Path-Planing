"""Load and validate ``configs/protocol_v2.yaml``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

DEFAULT_PATH = Path("configs") / "protocol_v2.yaml"
PLANNERS = ("astar", "apf", "apfe", "hybrid")
SCENARIO_IDS = {"open": 1, "concave_trap": 2, "ridge_corridor": 3, "high_density": 4}
APFE_KEYS = ("max_escapes", "turn_steps_min", "turn_steps_max", "straight_steps", "turn_frac")


class ProtocolError(ValueError):
    pass


def load(path: str | Path = DEFAULT_PATH) -> dict:
    """Return the protocol as a dict (plus ``_path`` and ``_sha256``)."""
    path = Path(path)
    raw = path.read_bytes()
    proto = yaml.safe_load(raw)
    proto["_path"] = str(path)
    proto["_sha256"] = hashlib.sha256(raw).hexdigest()
    _validate(proto)
    return proto


def _validate(p: dict) -> None:
    for key in ("status", "seeds", "monte_carlo", "resolutions_m", "altitude", "planners"):
        if key not in p:
            raise ProtocolError(f"protocol is missing '{key}'")
    if p["status"] not in ("draft", "registered"):
        raise ProtocolError("status must be 'draft' or 'registered'")
    if p["seeds"]["development"] == p["seeds"]["evaluation"]:
        raise ProtocolError("development and evaluation seeds must differ")
    if tuple(p["planners"]["order"]) != PLANNERS:
        raise ProtocolError(f"planner order must be {PLANNERS}")
    for name in p["monte_carlo"]["scenarios"]:
        if name not in SCENARIO_IDS or name not in p["scenarios"]:
            raise ProtocolError(f"unknown or unconfigured scenario {name!r}")
    if p["status"] == "registered":
        missing = [k for k in APFE_KEYS if p["planners"]["apfe"].get(k) is None]
        if missing:
            raise ProtocolError(f"registered protocol lacks APF-E parameters {missing}")


def apfe_params(p: dict) -> dict:
    """APF-E keyword arguments from the protocol (registered protocols only)."""
    a = p["planners"]["apfe"]
    if any(a.get(k) is None for k in APFE_KEYS):
        raise ProtocolError("APF-E parameters are not set (protocol still a draft?)")
    return {k: a[k] for k in APFE_KEYS}
