from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Union
import heapq
from utilities import get_action_cost, get_next_state
from railroad.core import State, Goal, Action, get_next_actions

# data structures for astar search
@dataclass
class Trajectory:
    """
    Data structure used to represent search tree trajectories (paths).
    """
    state_history: List[State]
    plan: List[Action]
    # used to avoid having to recompute the prob of no interruption for each child
    interruption_probs: List[float]
    cost: float = 0.0
    value: float = 0.0
    level: int = 0

    def create_child(
        self,
        goal: Goal,
        actions: List[Action],
        action: Action,
        interruption_value: float,
        interruption_prob: float,
        heuristic_fn: Union[float, Callable[[State, Goal, List[Action]], float]] = 0
    ) -> 'Trajectory':
        """
        Helper function for creation of trajectories on the frontier.
        """
        # compute f(n)
        accumulated_cost = discounted_accumulated_cost(
            self, action, interruption_value, interruption_prob
        )
        next_state, _ = get_next_state(self.state_history[-1], action)
        estimated_future_cost = h(self, action, goal, actions, heuristic_fn, interruption_prob)

        return Trajectory(
            cost=accumulated_cost,
            value=accumulated_cost+estimated_future_cost,
            level=self.level+1,
            state_history=self.state_history + [next_state],
            plan=self.plan + [action],
            interruption_probs=self.interruption_probs + [interruption_prob]
        )


def discounted_accumulated_cost(
    traj: Trajectory,
    action: Action,
    interruption_value: float,
    interruption_prob: float
) -> float:
    """
    Accumulated cost function of trajectory.
    """
    path_cost = traj.cost
    no_int_prob = get_no_int_prob(traj)

    # get reward
    r = get_action_cost(action)

    # discount the interruption value
    interruption_value*=interruption_prob

    return path_cost + no_int_prob * (r + interruption_value)


def h(
    traj: Trajectory,
    action: Action,
    goal: Goal,
    all_actions: List[Action],
    heuristic_fn: Union[int, float, Callable[[State, Goal, List[Action]], float]],
    next_interruption_prob: float
) -> float:
    """
    Heuristic function used to estimate the cost remaining for the trajectory.
    """
    if not isinstance(heuristic_fn, (int, float)):
        next_state, _ = get_next_state(traj.state_history[-1], action)
        estimated_q_value = heuristic_fn(next_state, goal, all_actions)
    else:
        estimated_q_value = heuristic_fn
    estimated_q_value *= (1 - next_interruption_prob)
    return get_no_int_prob(traj) * estimated_q_value


def astar_search(
    state: State,
    goal: Goal,
    actions: List[Action],
    interrupting_task_dist: Tuple[List[Goal], List[float]] | None,
    heuristic_fn: Union[int, float, Callable[[State, Goal, List[Action]], float]] = 0,
    interruption_prob_fn: Union[float, Callable[[State, Action], float]] = 0.1,
    num_steps: int = 1000
) -> Tuple[List[Action], float]:
    """
    Astar algorithm implementation.
    """
    value_cache: Dict[State, float] = dict()
    frontier = []

    # initial trajectory
    initial_traj = Trajectory(state_history=[state], plan=[], interruption_probs=[])
    heapq.heappush(frontier, (-1, initial_traj))

    # search loop
    for _ in range(num_steps):
        # find expansion node
        _, expand = heapq.heappop(frontier)
        curr_state = expand.state_history[-1]

        # check for goal condition being met
        if goal.evaluate(curr_state.fluents):
            return expand.plan, expand.cost

        # expand search tree
        available_actions = get_next_actions(curr_state, actions)
        for action in available_actions:
            # probability of interruption after taking action from current state
            next_state, interruption_prob = get_next_state(
                expand.state_history[-1],
                action,
                interruption_prob_fn
            )

            # value of next state for interrupting tasks needed and not found
            # compute and cache
            if interrupting_task_dist and not check_value_cache(next_state, value_cache):
                val = compute_interruption_value(
                    next_state, actions, interrupting_task_dist, heuristic_fn, interruption_prob_fn
                )
                value_cache[next_state] = val

            # construct new trajectory
            # use get method instead of directly indexing value_cache to account for case where
            # there are no interrupting tasks
            child_traj = expand.create_child(
                goal, actions, action, value_cache.get(next_state, 0),
                interruption_prob, heuristic_fn
            )
            q_value = child_traj.value
            heapq.heappush(frontier, (q_value, child_traj))

    # goal not reached, get best trajectory found
    _, best_found = heapq.heappop(frontier)
    return best_found.plan, best_found.cost


def compute_interruption_value(
    state: State,
    actions: List[Action],
    interrupting_task_dist: Tuple[List[Goal], List[float]],
    heuristic_fn: Union[int, float, Callable[[State, Goal, List[Action]], float]] = 0,
    interruption_prob_fn: Union[float, Callable[[State, Action], float]] = 0.1
) -> float:
    """
    Computes the expected value of a state for a task distribution.
    """
    expected_cost = 0.0
    for i in range(len(interrupting_task_dist[0])):
        task = interrupting_task_dist[0][i]
        prob = interrupting_task_dist[1][i]

        _, cost = astar_search(state, task, actions, None, heuristic_fn, interruption_prob_fn)
        expected_cost += (prob * cost)
    return expected_cost


def check_value_cache(state: State, value_cache: Dict[State, float]) -> bool:
    """
    Checks if the value of a state is already cached.
    """
    return True if value_cache.get(state) else False


def get_no_int_prob(traj: Trajectory) -> float:
    """
    Returns the probability of an interrupting task not arriving,
    based on the level of the search tree.
    """
    no_int_prob = 1
    for prob in traj.interruption_probs:
        no_int_prob*=(1 - prob)
    return no_int_prob
