import random
import csv
import contextlib
from functools import partial
from itertools import product
from collections import defaultdict
from typing import Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from railroad.core import (
    Fluent as F, State, det_ff_heuristic, Goal,
    get_action_by_name, convert_state_to_positive_preconditions
)
from railroad.environment import SymbolicEnvironment
from railroad.operators.core import construct_move_operator, construct_pick_operator, \
    construct_place_operator
from interruption_ap import astar_search
from utilities import (
    negative_fluent_preprocessing,
    construct_assemble_operator,
    get_augmented_task_dist,
    RandomVariableType,
    get_action_cost,
    get_task_arrival_prob,
    print_plan
)

# Set global font to serif family
plt.rcParams["font.family"] = "serif"


def main():
    """
    Run prototype experiments script.
    """
    # experiment settings
    output_fpath = "outputs/prototype_experimental_results.csv"
    baseline = [True, False]
    augment = [True, False]
    num_experiments = 100
    environment_interruption_probs = [0, 0.01, 0.05, 0.1, 0.12, 0.15, 0.20, 0.25, 0.3]
    random_seeds = [400, 270, 100, 140, 600, 499, 42, 82, 970]
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

    # get cartesian product of the experiment settings
    experiment_settings = product(
        augment,
        baseline,
        list(enumerate(interrupting_task_dists)),
        list(zip(environment_interruption_probs, random_seeds)),
    )

    # run experiments loop
    # store experimental results in the form:
    # experiment_num | augment | baseline | task_dist_idx | int_prob |
    # seed | total_cost | initial_plan
    results = []

    wrapped_experiment_settings = tqdm(experiment_settings, desc="Experiment Settings")
    for augment, baseline, (i, task_dist), (int_prob, seed) in wrapped_experiment_settings:
        # set seed. keep the seed the same for baseline/prototype, but different for
        # various interrupting task distributions and augment/replace.
        computed_seed = seed * (augment+i+1)
        random.seed(computed_seed)

        # augment the interrupting_task_dist if necessary
        if augment:
            task_dist = get_augmented_task_dist(goal, task_dist)
        for j in tqdm(range(num_experiments), desc="Experiments", leave=False):
            total_cost, plan, execution_trace = run_experiment(
                int_prob,
                goal,
                task_dist,
                baseline
            )

            results.append(
                [
                    j,
                    "augment" if augment else "replace",
                    "baseline" if baseline else "prototype",
                    i,
                    int_prob,
                    computed_seed,
                    total_cost,
                    plan,
                    execution_trace
                ]
            )

    # write out experimental results
    write_out_csv_results(
        output_fpath,
        ["ExperimentNum", "Augment", "Baseline", "Task_Dist_Idx", "Prob_Int", "Seed", "TotalCost"],
        [output[:-2] for output in results]
    )

    # write out plans and execution traces
    write_out_traces(results, mode="plan")
    write_out_traces(results, mode="execution")

    # write out summary of results
    results_summary = summarize_results(results)
    summary_out_fp = write_out_summary_for_viz(results_summary)

    # visualize results
    generate_plots(summary_out_fp)


def run_experiment(
    interruption_prob: float,
    goal: Goal,
    interrupting_task_dist: Tuple[List[Goal], List[float]],
    baseline_flag: bool
) -> Tuple[float, List[str] | None, List[str] | None]:
    """
    Runs a single experiment, given the parameters specified in main.
    Returns the total cost of the execution sequence, the initial plan, and a trace of 
    the execution sequence.
    """
    # the probability of interruption depends on the length of time required to execute an action
    interruption_prob_fn = partial(
        get_task_arrival_prob, RandomVariableType.CONTINUOUS, interruption_prob
    )

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
        planning_interruption_prob = interruption_prob_fn
        planning_interrupting_task_dist = converted_interrupting_task_dist
        current_task_reward = 0

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
        return -1, None, None

    # execution loop
    execution_trace = []

    for converted_action in plan:
        action = get_action_by_name(env.get_actions(), converted_action.name)
        # interrupting task arrived
        if goal.evaluate(env.state.fluents):
            execution_trace.append("Initial Goal Completed")
            break

        env.act(action)
        execution_trace.append(converted_action.name)

        if random.random() < interruption_prob_fn(get_action_cost(action)):
            execution_trace.append("New task arrived")
            break

    # re-plan for interrupting task that arrived or completion of current task
    interrupting_goal = interrupting_task_dist[0][0]
    converted_interrupting_goal = converted_interrupting_task_dist[0][0]
    converted_state = convert_state_to_positive_preconditions(env.state, mapping)

    re_plan, _ = astar_search(
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

    for converted_action in re_plan:
        action = get_action_by_name(env.get_actions(), converted_action.name)
        if interrupting_goal.evaluate(env.state.fluents):
            execution_trace.append("Interrupting Goal Completed")
            break
        env.act(action)
        execution_trace.append(converted_action.name)


    return (
        env.time if interrupting_goal.evaluate(env.state.fluents) else -1,
        [action.name for action in plan],
        execution_trace
    )


def setup_environment() -> SymbolicEnvironment:
    """
    Sets up the toy kitchen environment for prototyping experiments.
    """
    uniform_action_cost = 1

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

    def move_time(robot, loc_from, loc_to):
        return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))

    move = construct_move_operator(move_time)
    pick = construct_pick_operator(uniform_action_cost)
    place = construct_place_operator(uniform_action_cost)
    assemble = construct_assemble_operator(uniform_action_cost)

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


def write_out_csv_results(out_fpath: str, header: List[str], results: List) -> None:
    """
    Helper function for writing out experimental results to a
    single csv file.
    """
    if out_fpath.find("/") != -1:
        Path(out_fpath[:out_fpath.rindex("/")]).mkdir(parents=True, exist_ok=True)

    with open(out_fpath, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)


def write_out_traces(results: List[List], mode: str = "plan") -> None:
    """
    Helper function for writing out the computed plans
    from the experiments.
    """
    for num, aug, baseline, tdist_i, prob_int, _, _, plan, execution_trace in results:
        out_dir = f"plans/{aug}/task_dist_{tdist_i}/prob_int_{prob_int}/{baseline}"
        if mode == "plan":
            out_fpath = out_dir + f"/plan_{num}.txt"
        elif mode == "execution":
            out_fpath = out_dir + f"/execution_trace_{num}.txt"
        else: # invalid argument for mode provided
            return

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(out_fpath, 'w', encoding="utf-8") as fp:
            with contextlib.redirect_stdout(fp):
                print_plan(plan if mode == "plan" else execution_trace)


def summarize_results(results: List[List]) -> Dict[Tuple, float]:
    """
    Helper function for summarizing the experimental results.
    """
    grouped_results = defaultdict(list)
    for _, augment, baseline, task_dist_i, prob_int, _, total_cost, _, _ in results:
        grouped_results[(augment, baseline, task_dist_i, prob_int)].append(total_cost)
    return {k: sum(v) / len(v) for k, v in grouped_results.items()}


def write_out_summary_for_viz(summary: Dict[Tuple[str, str, int, float], float]) -> str:
    """
    Writes out a summary of the experimental results as a csv in the format
    expected by the visualization function.
    Returns the filepath of the written out summary file.
    """
    formatted_dict = defaultdict(list)
    for augment, _, task_dist_i, prob_int in summary.keys():
        if (augment, task_dist_i, prob_int) in formatted_dict:
            continue

        formatted_dict[(augment, task_dist_i, prob_int)].extend(
            [
                summary[(augment, "prototype", task_dist_i, prob_int)],
                summary[(augment, "baseline", task_dist_i, prob_int)]
            ]
        )

    formatted_summary = [
        [augment, task_dist_i, prob_int, p_cost, b_cost]
        for (augment, task_dist_i, prob_int), [p_cost, b_cost] in formatted_dict.items()
    ]

    out_fpath = "outputs/summary.csv"
    write_out_csv_results(
        out_fpath,
        ["Augment", "Task_Dist_Idx", "Prob_Int", "Prototype_AvgCost", "Baseline_AvgCost"],
        formatted_summary
    )
    return out_fpath


def plot(df: pd.DataFrame, out_fp: str) -> None:
    """
    Visualization function for comparing the average total costs
    of the baseline and proposed approach across various probabilities
    of task arrival.
    """
    # Set figure size: (width, height) in inches
    plt.figure(figsize=(8, 5))  # wide and short

    # Remove rows with missing values for series 1
    series1 = df[["Prob_Int", "Baseline_AvgCost"]].dropna()

    # Remove rows with missing values for series 2
    series2 = df[["Prob_Int", "Prototype_AvgCost"]].dropna()

    # Get colors from tab20c colormap
    cmap = plt.get_cmap("tab10")
    color1 = cmap(0)   # first color
    color2 = cmap(1)   # another distinct color

    # Plot first series
    plt.plot(
        series1["Prob_Int"],
        series1["Baseline_AvgCost"],
        marker='o',
        linestyle='-',
        color=color1,
        label='Baseline'
    )

    # Plot second series
    plt.plot(
        series2["Prob_Int"],
        series2["Prototype_AvgCost"],
        marker='s',
        linestyle='-',
        color=color2,
        label='Proposed Approach'
    )

    # Labels and title
    plt.xlabel("Probability of Interruption", fontsize=14)
    plt.ylabel("Total Cost", fontsize=14)

    # Set y-axis scale
    plt.ylim(0, 14)

    # Legend and grid
    plt.legend()

    # Save figure
    plt.savefig(out_fp, dpi=600, bbox_inches='tight')

    # Show plot
    # plt.show()


def generate_plots(summary_stats_fpath: str) -> None:
    """
    Helper function for generating the comparison plots between the performance
    of the baseline and the performance of the proposed approach across all
    experimental settings.
    """
    Path("plots/").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_stats_fpath)
    augment_vals = df["Augment"].unique()
    task_dist_idx_vals = df["Task_Dist_Idx"].unique()

    for augment, task_dist_idx in product(augment_vals, task_dist_idx_vals):
        plot(
            df.query("Augment == @augment and Task_Dist_Idx == @task_dist_idx"),
            f"plots/scatter_{augment}_taskdist_idx={task_dist_idx}.png"
        )


if __name__ == "__main__":
    main()
