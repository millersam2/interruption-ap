from typing import Callable, List, Union
from railroad.core import Action, Goal, State
from railroad.dashboard import PlannerDashboard


class AstarDashboardPlanner:
    """
    Adapts astar_search (a one-shot planner) to the DashboardPlanner protocol
    expected by railroad.dashboard.PlannerDashboard, which is otherwise built
    around incremental, per-step planners like MCTSPlanner.
    """

    def __init__(
        self,
        actions: List[Action],
        heuristic_fn: Union[int, float, Callable[[State, Goal, List[Action]], float]] = 0,
    ):
        self._actions = actions
        self._heuristic_fn = heuristic_fn

    def heuristic(self, state: State, goal: Goal) -> float:
        """
        Returns the value of the non-discounted heuristic function used as part
        of the Astar planning algorithm for a given state and the current task
        in a real-time task stream setting.
        """
        if isinstance(self._heuristic_fn, (int, float)):
            return float(self._heuristic_fn)
        return self._heuristic_fn(state, goal, self._actions)

    def get_trace_from_last_mcts_tree(self) -> str:
        """
        A placeholder function since astar_search has no tree to trace;
        nothing meaningful to report.
        """
        return ""


# TODO - need to read/test
def merge_dashboard_trajectories(dashboards: List[PlannerDashboard]) -> PlannerDashboard:
    """
    Consolidates the trajectory data (entity positions, nav positions, grid
    snapshots, history, and known robots) of multiple PlannerDashboard
    instances into the last dashboard in the list, so that show_plots() and
    save_video() render a single continuous trajectory spanning all of them.

    Assumes the dashboards share the same underlying env and were run back to
    back (i.e., their recorded timestamps are already mutually consistent).
    """
    target = dashboards[-1]
    for dash in dashboards[:-1]:
        for entity, positions in dash._entity_positions.items():
            target._entity_positions.setdefault(entity, [])
            target._entity_positions[entity] = sorted(
                target._entity_positions[entity] + positions, key=lambda p: p[0]
            )
        for robot, positions in dash._nav_continuous_positions.items():
            target._nav_continuous_positions.setdefault(robot, [])
            target._nav_continuous_positions[robot] = sorted(
                target._nav_continuous_positions[robot] + positions, key=lambda p: p[0]
            )
        target._nav_grid_snapshots = sorted(
            target._nav_grid_snapshots + dash._nav_grid_snapshots, key=lambda s: s[0]
        )
        target.known_robots |= dash.known_robots
        target.history = dash.history + target.history
    return target
