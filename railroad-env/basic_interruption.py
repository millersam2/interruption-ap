import numpy as np
from railroad.core import Fluent as F, get_action_by_name, State, Operator, Effect
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad.operators.core import construct_move_operator, construct_pick_operator, \
    construct_place_operator

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

def main(good_behavior: bool = False):
    """Basic Task Interruption Scenario.
        Predicates:
            at ?o - object ?l - location ; the object is at a location l
            hand-full ?r - robot ; the robot's gripper is holding an object
            holding ?r - robot ?o - object ; the robot is holding an object 
                                                in one of its grippers
            free ?r ; robot is free to take an action
            is-turkey ?o ; is the object turkey
            is-bread ?o ; is the object bread
            sandwhich-made ; sandwhich has been successfully made
            prep-station ?l ; location where the sandwhich can be made
    """
    # define operators
    def move_time(robot, loc_from, loc_to):
        return float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))

    move = construct_move_operator(move_time)
    pick = construct_pick_operator(PICK_TIME)
    place = construct_place_operator(PLACE_TIME)
    assemble = construct_assemble_operator(ASSEMBLE_TIME)

    # Setup task planning environment
    initial_state = construct_initial_state(good_behavior)
    env = SymbolicEnvironment(
        state=initial_state, objects_by_type=OBJECTS_BY_TYPE,
        operators=[move, pick, place, assemble],
    )

    total_time_steps = 20

    # Task 1: make sandwhich
    goal = F("sandwhich-made")

    with PlannerDashboard(goal, env) as dashboard:
        # Plan-act loop: replan whenever a robot becomes free
        for i in range(total_time_steps): # time steps
            if goal.evaluate(env.state.fluents) or i == 4:
                break

            actions = env.get_actions()
            planner = MCTSPlanner(actions, use_det_heuristic=True)
            action_name = planner(env.state, goal, max_iterations=10000, c=20)
            action = get_action_by_name(actions, action_name)
            env.act(action)
            dashboard.update(planner, action_name)

    # Interrupting Task: Clean countertop1
    goal = (
        ~F("at turkey countertop1") & ~F("at bread countertop1") &
        ~F("hand-full robot1") #& goal
    )

    with PlannerDashboard(goal, env) as dashboard:
        # Plan-act loop: replan whenever a robot becomes free
        for _ in range(total_time_steps): # time steps
            if goal.evaluate(env.state.fluents):
                break

            actions = env.get_actions()
            planner = MCTSPlanner(actions, use_det_heuristic=True)
            action_name = planner(env.state, goal, max_iterations=10000, c=20)
            action = get_action_by_name(actions, action_name)
            env.act(action)
            dashboard.update(planner, action_name)

    return 0

# helper functions
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

def construct_initial_state(good_behavior: bool = False) -> State:
    """
    Constructs the initial state of the environment.
    """
    # Initial state (s_0)
    initial_fluents = {
        F("free robot1"), F("at robot1 table"), F("is-turkey turkey"), F("is-bread bread"),
        ~F("hand-full robot1"), F("at turkey refrigerator"), F("at bread pantry"),
        ~F("prep-station table"), F("prep-station countertop2"), ~F("prep-station refrigerator"),
        ~F("sandwhich-made")
    }
    initial_fluents.add(
        ~F("prep-station countertop1") if good_behavior else F("prep-station countertop1")
    )
    return State(0.0, initial_fluents)

if __name__ == "__main__":
    main(good_behavior=False)
