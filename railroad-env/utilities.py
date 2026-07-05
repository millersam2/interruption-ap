from typing import List, Union, Callable, Tuple
from enum import Enum
from railroad.core import (
    Action, Goal, State, transition, Fluent as F, LiteralGoal, Operator, Effect
)
from railroad.core import (
    extract_negative_preconditions,
    extract_negative_goal_fluents,
    create_positive_fluent_mapping,
    convert_state_to_positive_preconditions,
    convert_action_to_positive_preconditions,
    convert_action_effects,
    convert_goal_to_positive_preconditions
)

# global constants/enums
class RandomVariableType(Enum):
    """
    Enumeration for the valid types of a random variable. Ensures that the 
    user can only pass in a valid random variable type for get_interruption_prob function.
    """
    DISCRETE = 1
    CONTINUOUS = 2


# utility functions for interruption anticipatory planning
def get_action_cost(action: Action) -> float:
    """
    Gets the total cost (reward) of performing an action.
    """
    return action.effects[-1].time + action.extra_cost


def get_next_state(
    state: State,
    action: Action,
    interrupting_prob_fn: Union[Callable[[float], float], float] = 0
) -> Tuple[State, float]:
    """
    Gets the state s' after performing an action a in s. This function
    assumes the state transition is deterministic. Additionally, returns
    the probability of an interrupting task arriving after the transition.
    """
    if isinstance(interrupting_prob_fn, (float, int)):
        interruption_prob = interrupting_prob_fn
    else:
        interruption_prob = interrupting_prob_fn(get_action_cost(action))
    outcomes = transition(state, action)
    assert len(outcomes) == 1
    next_state, prob = outcomes[0]
    assert prob == 1.0
    return next_state, interruption_prob


def negative_fluent_preprocessing(actions: List[Action], state: State, goals: List[Goal]):
    """
    Wrapper function to convert negative fluents to equivalent positive fluents. Important
    when using the FF heuristic.
    """
    # normalize goals if necessary
    goals = [LiteralGoal(g) if isinstance(g, F) else g for g in goals]

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


def construct_assemble_operator(assemble_time: int):
    """
    Constructs an assemble sandwhich operator.
    """
    assemble = Operator(
            name="assemble",
            parameters=[
                ("?r", "robot"), ("?o1", "object"), ("?o2", "object"),
                ("?o3", "object"), ("?l", "location")
            ],
            preconditions=[
                F("free ?r"), F("is-turkey ?o1"), F("is-bread ?o2"), F("at ?o1 ?l"),
                F("at ?o2 ?l"), F("at ?r ?l"), ~F("hand-full ?r"), F("prep-station ?l"),
                F("is-sandwhich ?o3")
            ],
            effects=[
                Effect(time=0, resulting_fluents={F("not free ?r"), F("hand-full ?r")}),
                Effect(time=assemble_time, resulting_fluents={
                    F("free ?r"), F("not at ?o1 ?l"), F("not at ?o2 ?l"),
                    F("sandwhich-made"), ~F("hand-full ?r"),
                    F("at ?o3 ?l")
                })
            ]
        )
    return assemble


def get_task_arrival_prob(
    rv_type: RandomVariableType,
    arrival_prob: float,
    action_time: float = -1
) -> float:
    """
    Helper function that returns the probability of a task arriving after the execution
    of an action. Supports both per-action (treating the random variable as discrete) and
    per-time-unit (treating the random variable as continuous) probabilities.
    """
    if rv_type == RandomVariableType.DISCRETE or action_time == -1:
        return arrival_prob
    return min(arrival_prob * action_time, 1.0)


def print_plan(actions: List[str]) -> None:
    """
    Helper function for printing out the best plan in a more
    readable format.
    """
    # print("Best Plan:")
    for i, action in enumerate(actions):
        print(f"{i}. {action}")


def get_augmented_task_dist(
    current_task: F | LiteralGoal,
    interrupting_task_dist: Tuple[List[Goal], List[float]]
) -> Tuple[List[Goal], List[float]]:
    """
    Helper function for task augmentation experiments. Given
    the passed in interrupting_task_dist, creates new future
    tasks that include the current task. Does not make changes
    in-place.
    """
    augmented_tasks = []
    probs = []
    for task, prob in zip(*interrupting_task_dist):
        augmented_tasks.append(current_task & task)
        probs.append(prob)
    return (augmented_tasks, probs)
