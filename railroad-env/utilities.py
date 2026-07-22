import json
import math
from typing import List, Union, Callable, Tuple, Set, Any, Dict
from enum import Enum
from railroad.core import (
    Action, Goal, State, transition, Fluent as F, LiteralGoal
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


class DistributionType(Enum):
    """
    Enumeration for the supported types of distributions for get_task_arrival_prob.
    """
    UNIFORM = 1
    EXPONENTIAL = 2


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


def get_task_arrival_prob(
    rv_type: RandomVariableType,
    arrival_prob: float,
    distribution_type: DistributionType | None = DistributionType.UNIFORM,
    time_for_prob: float = 100,
    action_time: float = -1,
) -> float:
    """
    Helper function that returns the probability of a task arriving after the execution
    of an action. Supports both per-action (treating the random variable as discrete) and
    per-time-unit (treating the random variable as continuous) probabilities.
    """
    if rv_type == RandomVariableType.DISCRETE or action_time == -1:
        return arrival_prob
    if (
        rv_type == RandomVariableType.CONTINUOUS and
        (
            distribution_type == DistributionType.UNIFORM or
            arrival_prob == 1
        )
    ):
        return min(arrival_prob * action_time, 1.0)
    # case: exponential distribution and arrival_prob != 1
    # arrival_prob is now parameter Beta for the exponential distribution
    beta = _calibrate_beta_parameter(arrival_prob, time_for_prob)
    return 1 - math.exp(-beta * action_time)


def _calibrate_beta_parameter(prob: float, a_t: float | int) -> float:
    """
    Helper function for computing the value of the beta parameter for the
    CDF of the exponential distribution such the provided time to complete
    an action will have the specified probability.
    Returns the computed Beta parameter when valid inputs provided
    (Prob: [0, 1) and a_t >= 0). Otherwise returns -1 on invalid inputs.
    """
    if prob < 0 or prob >= 1 or a_t < 0:
        return -1
    return -math.log(1 - prob) / a_t


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
    NOTE - currently this function doesn't provide any checking for
    tasks with conflicting goal fluents.
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


# helper functions for debugging/testing behavior in ProcTHOR environments
def handcrafted_interruption_value(prob_int: float, state_fluents: Tuple[F]) -> float:
    """
    Function used to test the source of the growing planning time required
    when transitioning to ProcTHOR environments.
    """
    good_fluent_sets = [
        {F("holding r1-left spoon_15")},#, F("holding r1-right pan_17")},
        {F("holding r1-right spoon_15")},#, F("holding r1-left pan_17")},
        {F("at pan_17 shelvingunit_6"), F("holding r1-left spoon_15")},
        {F("at pan_17 shelvingunit_6"), F("holding r1-right spoon_15")},
    ]
    good_state = bool([1 for fs in good_fluent_sets if fs.issubset(state_fluents)])
    if prob_int >= 0.1 and good_state:
        return -500
    return 500
