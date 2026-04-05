from dataclasses import dataclass
from typing import List
from railroad.core import Action, Goal, State, transition
from railroad.core import (
    extract_negative_preconditions,
    extract_negative_goal_fluents,
    create_positive_fluent_mapping,
    convert_state_to_positive_preconditions,
    convert_action_to_positive_preconditions,
    convert_action_effects,
    convert_goal_to_positive_preconditions
)

# utility functions and classes for interruption anticipatory planning
@dataclass
class Trajectory:
    """
    Data structure used to represent search tree trajectories (paths).
    """
    state_history: List[State]
    plan: List[Action]
    cost: float = 0.0
    value: float = 0.0
    level: int = 0


def get_action_cost(action: Action) -> float:
    """
    Gets the total cost (reward) of performing an action.
    """
    return action.effects[-1].time + action.extra_cost


def get_next_state(state: State, action: Action) -> State:
    """
    Gets the state s' after performing an action a in s. This function
    assumes the state transition is deterministic.
    """
    outcomes = transition(state, action)
    assert len(outcomes) == 1
    next_state, prob = outcomes[0]
    assert prob == 1.0
    return next_state


def negative_fluent_preprocessing(actions: List[Action], state: State, goals: List[Goal]):
    """
    Wrapper function to convert negative fluents to equivalent positive fluents. Important
    when using the FF heuristic.
    """
    # build negative fluent to equivalent positive fluent mapping
    negative_preconditions = extract_negative_preconditions(actions)
    for goal in goals:
        negative_preconditions = negative_preconditions | extract_negative_goal_fluents(goal)
    mapping = create_positive_fluent_mapping(negative_preconditions)

    # convert actions using mapping
    converted_actions = []
    for action in actions:
        action_pos_precond = convert_action_to_positive_preconditions(action, mapping)
        converted_action = convert_action_effects(action_pos_precond, mapping)
        converted_actions.append(converted_action)

    # convert state using mapping
    converted_state = convert_state_to_positive_preconditions(state, mapping)

    # convert goals using mapping
    converted_goals = []
    for goal in goals:
        converted_goal = convert_goal_to_positive_preconditions(goal, mapping)
        converted_goals.append(converted_goal)
    return converted_actions, converted_state, converted_goals, mapping
