from typing import List, Union, Callable, Tuple
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

# utility functions for interruption anticipatory planning
def get_action_cost(action: Action) -> float:
    """
    Gets the total cost (reward) of performing an action.
    """
    return action.effects[-1].time + action.extra_cost


def get_next_state(
    state: State,
    action: Action,
    interrupting_prob_fn: Union[Callable[[State, Action], float], float] = 0
) -> Tuple[State, float]:
    """
    Gets the state s' after performing an action a in s. This function
    assumes the state transition is deterministic. Additionally, returns
    the probability of an interrupting task arriving after the transition.
    """
    if isinstance(interrupting_prob_fn, (float, int)):
        interruption_prob = interrupting_prob_fn
    else:
        interruption_prob = interrupting_prob_fn(state, action)
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
                ("?r", "robot"), ("?o1", "object"), ("?o2", "object"), ("?l", "location")
            ],
            preconditions=[
                F("free ?r"), F("is-turkey ?o1"), F("is-bread ?o2"), F("at ?o1 ?l"),
                F("at ?o2 ?l"), F("at ?r ?l"), ~F("hand-full ?r"), F("prep-station ?l")
            ],
            effects=[
                Effect(time=0, resulting_fluents={F("not free ?r"), F("hand-full ?r")}),
                Effect(time=assemble_time, resulting_fluents={
                    F("free ?r"), F("not at ?o1 ?l"), F("not at ?o2 ?l"),
                    F("sandwhich-made"), ~F("hand-full ?r")
                })
            ]
        )
    return assemble
