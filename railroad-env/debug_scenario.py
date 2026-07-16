from typing import Tuple, List, Set, Dict
from functools import partial
import numpy as np
from railroad.core import Fluent as F, State, ff_heuristic, Goal, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.operators.core import construct_move_operator, construct_pick_operator, \
    construct_place_operator
from railroad.dashboard import PlannerDashboard
from environments import KitchenProcTHOREnvironment
from interruption_ap import astar_search
from utilities import (
    negative_fluent_preprocessing, construct_assemble_operator,
    get_task_arrival_prob, RandomVariableType, print_plan
)
from dashboard_adapters import AstarDashboardPlanner

def main():
    """Debugging scenario for interruption ap planner, where the planner
    takes into account the possiblity of interrupting tasks arriving, but the 
    environment does not allow interrupting tasks to arrive.
    """
    # parameters
    prob_int = 0
    interruption_replaces = True # False signifies that the interruption should augment
    procthor_environment = True # False signifies the simple prototype environment
    seed = 201
    name = f"procthor-{"replace" if interruption_replaces else "augment"}-p={prob_int}-{seed}"
    save_plot = f"{name}.jpg"
    save_video = f"{name}.mp4"
    show_plot = False

    if not procthor_environment:
        env = construct_simple_kitchen_environment()
        goal = get_simple_goal()
        interrupting_task_dist = get_simple_task_distribution(interruption_replaces)
    else:
        env = construct_procthor_kitchen_environment(seed)
        goal = get_example_procthor_goal()
        interrupting_task_dist = get_example_procthor_task_distribution(interruption_replaces)

    # fluent pre-processing for usage of FF heuristic
    all_goals = [goal] + [goal for goal in interrupting_task_dist[0]]
    actions, initial_state, converted_goals, _= negative_fluent_preprocessing(
        env.get_actions(), env.state, all_goals
    )

    goal = converted_goals[0]
    interrupting_task_dist = (list(converted_goals[1:]), interrupting_task_dist[1])

    task_arrival_prob_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS, prob_int
    )

    plan, cost = astar_search(
        initial_state,
        goal,
        actions,
        None,
        ff_heuristic,
        task_arrival_prob_fn,
        0,
        num_steps=100000,
        print_trace=False
    )

    if not procthor_environment:
        # temporary, very basic plan outputs for debugging
        action_names = [action.name for action in plan]
        print(f"Probability of Interruption : {prob_int}")
        print_plan(action_names)
        print(f"Discounted Plan Cost: {cost}")
    else: # for procthor_environments
        dash_env = construct_procthor_kitchen_environment(seed)
        planner_factory = partial(AstarDashboardPlanner, heuristic_fn=ff_heuristic)
        with PlannerDashboard(goal, dash_env, planner_factory=planner_factory) as dashboard:
            adapter = AstarDashboardPlanner(dash_env.get_actions(), ff_heuristic)
            for converted_action in plan:
                action = get_action_by_name(dash_env.get_actions(), converted_action.name)
                dash_env.act(action)
                dashboard.update(adapter, action.name)

        dashboard.show_plots(
            save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        )


# helper functions
def construct_simple_kitchen_environment() -> SymbolicEnvironment:
    """
    Constructs a SymbolicEnvironment representing the simple kitchen
    environment used for prototyping.
    """
    locations = {
        "refrigerator": np.array([0, 0]),
        "pantry": np.array([1, 0]),
        "countertop1": np.array([1,1]),
        "countertop2": np.array([2,1]),
        "table": np.array([0,2])
    }

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(locations),
        "object": {"turkey", "bread", "sandwhich"}
    }

    pick_time = 1
    place_time =1
    assemble_time = 1

    # define operators
    def move_time(robot, loc_from, loc_to):
        return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))

    move = construct_move_operator(move_time)
    pick = construct_pick_operator(pick_time)
    place = construct_place_operator(place_time)
    assemble = construct_assemble_operator(assemble_time)

    # initial state
    initial_fluents = {
        F("free robot1"), F("at robot1 table"), F("is-turkey turkey"), F("is-bread bread"),
        ~F("hand-full robot1"), F("at turkey refrigerator"), F("at bread pantry"),
        ~F("prep-station table"), F("prep-station countertop2"), ~F("prep-station refrigerator"),
        ~F("sandwhich-made"), F("prep-station countertop1"), F("is-sandwhich sandwhich"),
        ~F("prep-station pantry")
    }
    initial_state = State(0.0, initial_fluents)

    env = SymbolicEnvironment(
        state=initial_state, objects_by_type=objects_by_type,
        operators=[move, pick, place, assemble],
    )
    return env


def get_simple_task_distribution(
    interruption_replaces: bool = True
) -> Tuple[List[Goal], List[float]]:
    """
    Returns the task distribution for the prototype kitchen scenario.
    """
    # Task replacement
    replace_task_dist = (
        [
            (
                ~F("at turkey countertop1") & ~F("at bread countertop1") &
                ~F("hand-full robot1") & ~F("at sandwhich countertop1")
            )
        ],
        [1.0]
    )

    # Task augmentation
    augment_task_dist = (
        [
            (
                ~F("at turkey countertop1") & ~F("at bread countertop1") &
                ~F("hand-full robot1") & ~F("at sandwhich countertop1") &
                F("sandwhich-made")
            )
        ],
        [1.0]
    )

    return replace_task_dist if interruption_replaces else augment_task_dist


def get_simple_goal() -> F | Goal:
    """
    Gets the initial goal for the prototype kitchen scenario.
    """
    return (F("sandwhich-made") & F("at sandwhich table"))


def construct_procthor_kitchen_environment(seed: int) -> KitchenProcTHOREnvironment:
    """
    Constructs a KitchenProcTHOREnvironment representing the scene 
    corresponding to the scene from ProcTHOR-10k.
    """
    initial_fluents = {
        F("at robot1 start_loc"), F("free robot1"),
        F("gripper-of r1-left robot1"), F("gripper-of r1-right robot1"),
        ~F("hand-full r1-left"), ~F("hand-full r1-right")
    }
    initial_state = State(0.0, initial_fluents)

    env = KitchenProcTHOREnvironment(
        seed,
        initial_state,
        {
            "robot": {"robot1"},
            "gripper": {"r1-left", "r1-right"},
            "location": {"start_loc"},
        }
    )

    # Fully populate symbolic environment now that scene is available internally.
    env.objects_by_type["location"] = set(env.scene.locations.keys())
    env.objects_by_type["object"] = env.scene.objects
    setup_procthor_initial_state(env.fluents, env.scene.object_locations)
    return env


def setup_procthor_initial_state(
    initial_fluents: Set[F],
    objects_by_location: Dict[str, Set]
) -> None:
    """
    Helper function that adds fluents related to the location of 
    objects (e.g., at ?obj ?loc) to the initial fluents of a
    KitchenProcTHOREnvironment.
    """
    object_location_fluents = [
        F(f"at {obj} {loc}")
        for loc in objects_by_location
        for obj in objects_by_location[loc]
    ]
    return initial_fluents.update(object_location_fluents)


def get_example_procthor_task_distribution(
    interruption_replaces: bool = True
) -> Tuple[List[Goal], List[float]]:
    """
    Returns the task distribution for an example ProcTHOR kitchen scenario
    (seed=201).
    """
    # Task replacement
    replace_task_dist = (
        [
            (
                F("at spoon_15 shelvingunit_6")
            )
        ],
        [1.0]
    )

    # Task augmentation
    augment_task_dist = (
        [
            (
                F("at spoon_15 shelvingunit_6") & F("at pan_17 shelvingunit_6")
            )
        ],
        [1.0]
    )

    return replace_task_dist if interruption_replaces else augment_task_dist


def get_example_procthor_goal() -> F | Goal:
    """
    Gets the initial goal for an example ProcTHOR kitchen scenario (seed=201).
    """
    return F("at pan_17 shelvingunit_6")


if __name__ == "__main__":
    main()
