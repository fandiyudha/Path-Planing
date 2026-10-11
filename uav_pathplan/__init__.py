"""Risk-aware UAV path-planning simulation framework.

Compares A*, Artificial Potential Field (APF), and a Hybrid A*-APF planner
in a 2.5D environment with radar threats modulated by terrain masking.

All three planners read the same elevation field and the same risk field, and
every resulting path is evaluated with the same cost function J
(:func:`uav_pathplan.cost.evaluate`).  The planners' own objectives differ: A*
minimises an additive edge cost, APF follows a potential-field gradient, and the
Hybrid combines both.  See ``config.Config`` for every parameter.
"""

__all__ = ["config", "terrain"]
