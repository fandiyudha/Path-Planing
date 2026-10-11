"""Path planners: A*, APF, and Hybrid A*-APF.

All three minimise the SAME cost function (``uav_pathplan.cost``) on the SAME
risk field with the SAME altitude policy.  Only the search mechanism differs.
Each planner returns a :class:`PlanResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlanResult:
    """Uniform result container for every planner."""

    name: str
    path: np.ndarray  # (M, 2) world-metre waypoints (empty if fail)
    success: bool
    time_s: float  # wall-clock planning time (s)
    n_exp: int = 0  # A*: expanded nodes
    n_iter: int = 0  # APF/Hybrid: gradient iterations
    info: dict = field(default_factory=dict)  # planner-specific extras

    @property
    def n_waypoints(self) -> int:
        return 0 if self.path is None else int(np.asarray(self.path).shape[0])


__all__ = ["PlanResult"]
