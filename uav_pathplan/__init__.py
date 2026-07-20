"""Risk-aware UAV path-planning simulation framework.

Compares A*, Artificial Potential Field (APF), and a Hybrid A*-APF planner
in a 2.5D environment with radar threats modulated by terrain masking.

All three planners read the same elevation field, the same risk field, and
minimize the same cost function J. See ``config.Config`` for every parameter.
"""

__all__ = ["config", "terrain"]
