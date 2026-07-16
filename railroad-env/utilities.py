import json
from typing import List, Union, Callable, Tuple, Set, Any, Dict
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
from railroad.operators._utils import OptNumeric, _to_numeric
from railroad.environment.procthor.resources import get_procthor_10k_dir
from railroad.environment.procthor.utils import get_generic_name

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


# custom operator functions
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


def construct_gripper_pick_operator(pick_time: OptNumeric) -> Operator:
    """Construct a basic pick operator (non-blocking).

    Args:
        pick_time: Time or function for pick duration.
            Function signature: (robot, gripper, location, object) -> float

    Returns:
        Operator for picking up an object.
    """
    pick_time_fn = _to_numeric(pick_time)
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?g", "gripper"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?g")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(
                time=(pick_time_fn, ["?r", "?g", "?loc", "?obj"]),
                resulting_fluents={F("free ?r"), F("holding ?g ?obj"), F("hand-full ?g")},
            ),
        ],
    )


def construct_gripper_place_operator(place_time: OptNumeric) -> Operator:
    """Construct a basic place operator (non-blocking).

    Args:
        place_time: Time or function for place duration.
            Function signature: (robot, gripper, location, object) -> float

    Returns:
        Operator for placing an object.
    """
    place_time_fn = _to_numeric(place_time)
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?g", "gripper"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?g ?obj"), F("hand-full ?g")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?g ?obj")}),
            Effect(
                time=(place_time_fn, ["?r", "?g", "?loc", "?obj"]),
                resulting_fluents={F("free ?r"), F("at ?obj ?loc"), ~F("hand-full ?g")},
            ),
        ],
    )


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


# helper functions for ProcTHOR-10k dataset experiments
def filter_procthor_scenes(
    num_rooms: Set[int] | None = None,
    room_types: Set[str] | None = None,
    locations: Set[str] | None = None,
    objects: Set[str] | None = None
) -> List[int]:
    """
    Filters the scenes of the ProcTHOR-10k dataset based on 
    the number of rooms in the scene, if 1 or more of the rooms
    in the scene have the desired roomType, and/or the scene contains
    user-specified locations (containers) and objects, which must be
    lowercase strings.
    Returns a list of the indicies of scenes (seeds) that have the 
    desired criteria.
    """
    # load in scene representations of ProcTHOR-10k
    data_dir = get_procthor_10k_dir()
    with open(data_dir / 'data.jsonl', 'r', encoding="utf-8") as f:
        json_list = list(f)

    # when no filter criteria are provided, return a list of all the seeds
    if (
        num_rooms is None and
        room_types is None and
        locations is None and
        objects is None
    ):
        return list(range(len(json_list)))

    filtered_scene_seeds = []
    for seed, scene_json in enumerate(json_list):
        scene = json.loads(scene_json)
        rooms = scene["rooms"]
        containers = scene["objects"]
        if (
            _check_num_rooms(rooms, num_rooms) and
            _check_scene_room_types(rooms, room_types) and
            _check_scene_locations(containers, locations) and
            _check_scene_objects(containers, objects)
        ):
            filtered_scene_seeds.append(seed)

    return filtered_scene_seeds


def _check_num_rooms(rooms: List[Dict[str, Any]], num_rooms: Set[int] | None) -> bool:
    """
    Helper function for checking if the number of rooms in a ProcTHOR scene
    matches the desired number of rooms.
    Returns True if the user doesn't specify the desired number of rooms or
    if a match is found. Otherwise, returns False.
    """
    return len(rooms) in num_rooms if num_rooms is not None else True


def _check_scene_room_types(rooms: List[Dict[str, Any]], room_types: Set[str] | None) -> bool:
    """
    Helper function for checking if 1 or more of the rooms in a ProcTHOR
    scene is of the desired type. (E.g., kitchen, bedroom, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if a match is found. Otherwise, returns False.
    """
    if room_types is None:
        return True
    return bool([True for room in rooms if room["roomType"] in room_types])


def _check_scene_locations(containers: List[Dict[str, Any]], locations: Set[str] | None) -> bool:
    """
    Helper function for checking if the ProcTHOR scene contains the 
    desired locations. (E.g., countertop, fridge, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if all locations are present. Otherwise, returns False.
    """
    if locations is None:
        return True
    scene_locations = {get_generic_name(container["id"]) for container in containers}
    return locations.issubset(scene_locations)


def _check_scene_objects(containers: List[Dict[str, Any]], objects: Set[str] | None) -> bool:
    """
    Helper function for checking if the ProcTHOR scene contains the
    desired objects. (E.g., coffeemachine, egg, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if all objects are present. Otherwise, returns False.
    """
    if objects is None:
        return True
    scene_objects = {
        get_generic_name(child["id"])
        for container in containers
        for child in container.get("children", [])
    }
    return objects.issubset(scene_objects)
