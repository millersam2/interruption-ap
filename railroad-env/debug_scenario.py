import numpy as np
from railroad.core import Fluent as F, get_action_by_name, State, Operator, Effect, det_ff_heuristic
from railroad.environment import SymbolicEnvironment
# from railroad.planner import MCTSPlanner
# from railroad.dashboard import PlannerDashboard
from railroad.operators.core import construct_move_operator, construct_pick_operator, \
    construct_place_operator
from interruption_ap import astar_search
from utilities import negative_fluent_preprocessing, construct_assemble_operator

# global constants
LOCATIONS = {
        "refrigerator": np.array([0, 0]),
        "pantry": np.array([1, 0]),
        "countertop1": np.array([1,1]),
        "countertop2": np.array([2,2]),
        "table": np.array([0,2])
}

OBJECTS_BY_TYPE = {
    "robot": {"robot1"},
    "location": set(LOCATIONS),
    "object": {"turkey", "bread"}
}

PICK_TIME = 1
PLACE_TIME = 1
ASSEMBLE_TIME = 3

def main():
    """Debugging scenario for interruption ap planner, where the planner
    takes into account the possiblity of interrupting tasks arriving, but the 
    environment does not allow interrupting tasks to arrive.
    """
    # define operators
    def move_time(robot, loc_from, loc_to):
        return float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))

    move = construct_move_operator(move_time)
    pick = construct_pick_operator(PICK_TIME)
    place = construct_place_operator(PLACE_TIME)
    assemble = construct_assemble_operator(ASSEMBLE_TIME)

    # Setup task planning environment
    initial_state = construct_initial_state()

    env = SymbolicEnvironment(
        state=initial_state, objects_by_type=OBJECTS_BY_TYPE,
        operators=[move, pick, place, assemble],
    )

    # Task: make sandwhich
    goal = F("sandwhich-made")

    # Interrupting Task Distribution: Clean-off countertop1
    interrupting_task_dist = (
        [
            (~F("at turkey countertop1") & ~F("at bread countertop1") & ~F("hand-full robot1"))
        ],
        [1.0]
    )

    # fluent pre-processing for usage of FF heuristic
    all_goals = [goal] + [goal for goal in interrupting_task_dist[0]]
    actions, initial_state, converted_goals, _= negative_fluent_preprocessing(
        env.get_actions(), initial_state, all_goals
    )

    goal = converted_goals[0]
    interrupting_task_dist = (list(converted_goals[1:]), interrupting_task_dist[1])

    plan, cost = astar_search(
        initial_state,
        goal,
        actions,
        interrupting_task_dist,
        det_ff_heuristic,
        0.1,
        # num_steps=10
        num_steps=1000000,
        print_trace=True
    )

    # temporary, very basic plan outputs for debugging
    action_names = [action.name for action in plan]
    print(f"Best Plan: {action_names}")
    print(f"Discounted Plan Cost: {cost}")


# helper functions
def construct_initial_state() -> State:
    """
    Constructs the initial state of the environment.
    """
    # Initial state (s_0)
    initial_fluents = {
        F("free robot1"), F("at robot1 table"), F("is-turkey turkey"), F("is-bread bread"),
        ~F("hand-full robot1"), F("at turkey refrigerator"), F("at bread pantry"),
        ~F("prep-station table"), F("prep-station countertop2"), ~F("prep-station refrigerator"),
        ~F("sandwhich-made"), F("prep-station countertop1")
    }
    return State(0.0, initial_fluents)


if __name__ == "__main__":
    main()
