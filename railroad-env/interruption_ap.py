from typing import Dict, Tuple, List, Callable, Union
import heapq
from utilities import Trajectory, get_action_cost, get_next_state
from railroad.core import State, Goal, Action, det_ff_heuristic, get_next_actions

# constants
INTERRUPTION_PROB = 0.1

def g(traj: Trajectory, action: Action, value_cache: Dict[State, float]) -> float:
    """
    Accumulated cost function of trajectory.
    """
    path_cost = traj.cost
    no_int_prob = get_no_int_prob(traj)

    # get reward
    r = get_action_cost(action)

    # get expected value of interruption
    next_state = get_next_state(traj.state_history[-1], action)
    interruption_value = value_cache.get(next_state, 0) * INTERRUPTION_PROB

    return path_cost + no_int_prob * (r + interruption_value)


def h(
    traj: Trajectory,
    action: Action,
    goal: Goal,
    all_actions: List[Action],
    heuristic_fn: Union[int, float, Callable[..., float]]
) -> float:
    """
    Heuristic function used to estimate the cost remaining for the trajectory.
    """
    if not isinstance(heuristic_fn, (int, float)):
        next_state = get_next_state(traj.state_history[-1], action)
        estimated_q_value = heuristic_fn(next_state, goal, all_actions)
    else:
        estimated_q_value = heuristic_fn
    estimated_q_value *= (1 - INTERRUPTION_PROB)
    return get_no_int_prob(traj) * estimated_q_value


def astar_search(
    meta_state: Tuple[State, Goal],
    actions: List[Action],
    interrupting_task_dist: Tuple[List[Goal], List[float]] | None,
    heuristic_fn: Union[int, float, Callable[..., float]] = 0,
    num_steps: int = 1000
) -> Tuple[List[Action], float]:
    """
    Astar algorithm implementation.
    """
    value_cache: Dict[State, float] = dict()
    frontier = []
    goal = meta_state[1]

    # initial trajectory
    initial_traj = Trajectory(state_history=[meta_state[0]], plan=[])
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
            next_state = get_next_state(expand.state_history[-1], action)

            # value of next state for interrupting tasks needed and not found
            # compute and cache
            if interrupting_task_dist and not check_value_cache(next_state, value_cache):
                val = compute_interruption_value(
                    next_state, actions, interrupting_task_dist, heuristic_fn
                )
                value_cache[next_state] = val

            # construct new trajectory
            child_traj, q_value= construct_trajectory(goal, actions, expand, action, value_cache)
            heapq.heappush(frontier, (q_value, child_traj))

    # goal not reached, get best trajectory found
    _, best_found = heapq.heappop(frontier)
    return best_found.plan, best_found.cost


def construct_trajectory(
    goal: Goal,
    actions: List[Action],
    parent: Trajectory,
    action: Action,
    value_cache: Dict[State, float],
    heuristic_fn: Union[int, float, Callable[..., float]] = 0
) -> Tuple[Trajectory, float]:
    """
    Helper function for creation of trajectories on the frontier.
    """
    # compute f(n)
    accumulated_cost = g(parent, action, value_cache)
    next_state = get_next_state(parent.state_history[-1], action)
    estimated_future_cost = h(parent, action, goal, actions, heuristic_fn)

    traj = Trajectory(
        cost=accumulated_cost,
        value=accumulated_cost+estimated_future_cost,
        level=parent.level+1,
        state_history=parent.state_history + [next_state],
        plan=parent.plan + [action]
    )
    return traj, traj.value


def compute_interruption_value(
    state: State,
    actions: List[Action],
    interrupting_task_dist: Tuple[List[Goal], List[float]],
    heuristic_fn: Union[int, float, Callable[..., float]] = 0
):
    """
    Computes the expected value of a state for a task distribution.
    """
    expected_cost = 0.0
    for i in range(len(interrupting_task_dist[0])):
        task = interrupting_task_dist[0][i]
        prob = interrupting_task_dist[1][i]

        _, cost = astar_search((state, task), actions, None, heuristic_fn)
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
    return (1 - INTERRUPTION_PROB) ** traj.level


def main():
    """
    Anticipatory Planning for scenarios with interrupting tasks
    """
    interrupting_task_distribution = []
    
    return


if __name__ == "__main__":
    main()
