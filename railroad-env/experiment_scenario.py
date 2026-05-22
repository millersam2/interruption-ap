import random
import csv
from typing import Tuple, List
from railroad.core import (
    Fluent as F, State, det_ff_heuristic, Goal,
    get_action_by_name, convert_state_to_positive_preconditions
)
from railroad.environment import SymbolicEnvironment
from railroad.operators.core import construct_move_operator, construct_pick_operator, \
    construct_place_operator
from interruption_ap import astar_search
from utilities import negative_fluent_preprocessing, construct_assemble_operator


def main():
    # experiment settings
    num_experiments = 30
    environment_interruption_probs = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3]
    random_seeds = [400, 270, 100, 140, 600, 499, 42]
    goal = F("sandwhich-made")
    interrupting_task_dists = [
        (
            [
                (
                    ~F("at turkey countertop1") & ~F("at bread countertop1") &
                    ~F("hand-full robot1") & ~F("at sandwhich countertop1")
                )
            ],
            [1.0]
        ),
        (
        [
            (
                ~F("at turkey countertop2") & ~F("at bread countertop2") &
                ~F("hand-full robot1") & ~F("at sandwhich countertop2")
            )
        ],
        [1.0]
        )
    ]

    # run experiments
    for baseline in [True, False]:
        for i, interrupting_task_dist in enumerate(interrupting_task_dists):
            for k, int_prob in enumerate(environment_interruption_probs):
                random.seed(random_seeds[k] * i+1)

                # interruption planner doesn't currently find a solution for this case
                if not baseline and int_prob == 0.3:
                    continue

                results = []
                for j in range(num_experiments):
                    total_cost = run_experiment(
                        int_prob,
                        goal,
                        interrupting_task_dist,
                        baseline
                    )
                    results.append([j, total_cost])

                # write out results
                if baseline:
                    output_filepath = f"outputs/baseline-results_int_task={i+1}_prob={int_prob}.csv"
                else:
                    output_filepath = f"outputs/results_int_task={i+1}_prob={int_prob}.csv"

                with open(output_filepath, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Experiment Number", "Total Cost"])
                    writer.writerows(results)


def run_experiment(
    interruption_prob: float,
    goal: Goal,
    interrupting_task_dist: Tuple[List[Goal], List[float]],
    baseline_flag: bool
) -> float:
    env = setup_environment()

    # fluent pre-processing for usage of FF heuristic
    all_goals = [goal] + [interrupting_goal for interrupting_goal in interrupting_task_dist[0]]
    converted_actions, initial_state, converted_goals, mapping= negative_fluent_preprocessing(
        env.get_actions(), env.state, all_goals
    )

    converted_goal = converted_goals[0]
    converted_interrupting_task_dist = (list(converted_goals[1:]), interrupting_task_dist[1])

    if baseline_flag:
        planning_interruption_prob = 0
        planning_interrupting_task_dist = None
        current_task_reward = 0
    else:
        planning_interruption_prob = interruption_prob
        planning_interrupting_task_dist = converted_interrupting_task_dist
        current_task_reward = -15

    plan, _ = astar_search(
        initial_state,
        converted_goal,
        converted_actions,
        planning_interrupting_task_dist,
        det_ff_heuristic,
        planning_interruption_prob,
        current_task_reward,
        num_steps=300000,
        print_trace=False
    )

    # check if the plan was successful
    if "assemble" not in plan[-1].name:
        return -1

    # execution loop
    for converted_action in plan:
        action = get_action_by_name(env.get_actions(), converted_action.name)
        # interrupting task arrived
        if goal.evaluate(env.state.fluents):
            break

        env.act(action)

        if random.random() < interruption_prob:
            break

    # re-plan for interrupting task that arrived or completion of current task
    interrupting_goal = interrupting_task_dist[0][0]
    converted_interrupting_goal = converted_interrupting_task_dist[0][0]
    converted_state = convert_state_to_positive_preconditions(env.state, mapping)

    plan, _ = astar_search(
        converted_state,
        converted_interrupting_goal,
        converted_actions,
        None,
        det_ff_heuristic,
        0,
        0,
        num_steps=300000,
        print_trace=False
    )

    for converted_action in plan:
        action = get_action_by_name(env.get_actions(), converted_action.name)
        if interrupting_goal.evaluate(env.state.fluents):
            break
        env.act(action)

    return env.time if interrupting_goal.evaluate(env.state.fluents) else -1


def setup_environment() -> SymbolicEnvironment:
    uniform_action_cost = 1

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"refrigerator", "pantry", "countertop1", "countertop2", "table"},
        "object": {"turkey", "bread", "sandwhich"}
    }

    move = construct_move_operator(uniform_action_cost)
    pick = construct_pick_operator(uniform_action_cost)
    place = construct_place_operator(uniform_action_cost)
    assemble = construct_assemble_operator(uniform_action_cost)

    initial_fluents = {
        F("free robot1"), F("at robot1 table"), F("is-turkey turkey"), F("is-bread bread"),
        ~F("hand-full robot1"), F("at turkey refrigerator"), F("at bread pantry"),
        ~F("prep-station table"), F("prep-station countertop2"), ~F("prep-station refrigerator"),
        ~F("sandwhich-made"), F("prep-station countertop1"), F("is-sandwhich sandwhich")
    }
    initial_state = State(0.0, initial_fluents)

    env = SymbolicEnvironment(
        state=initial_state, objects_by_type=objects_by_type,
        operators=[move, pick, place, assemble],
    )
    return env


if __name__ == "__main__":
    main()
