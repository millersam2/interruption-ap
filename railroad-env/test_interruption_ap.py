import pytest
from railroad.core import State, Fluent, get_next_actions, det_ff_heuristic
from railroad.operators.core import (
    construct_move_operator,
    construct_pick_operator,
    construct_place_operator
)
from interruption_ap import (
    check_value_cache,
    get_no_int_prob,
    discounted_accumulated_cost,
    h,
    astar_search,
    compute_interruption_value,
    Trajectory
)
from utilities import get_next_state

INTERRUPTION_PROB = 0.1

def test_check_value_cache():
    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    value_cache = {}
    assert not check_value_cache(initial_state, value_cache)

    value_cache[initial_state] = 2.5
    assert check_value_cache(initial_state, value_cache)


def test_get_no_int_prob():
    traj = Trajectory(state_history=[], plan=[], interruption_probs=[])
    assert get_no_int_prob(traj) == 1
    traj.interruption_probs.append(0.1)
    assert get_no_int_prob(traj) == 0.9
    traj.interruption_probs.append(0.1)
    assert get_no_int_prob(traj) == 0.81


@pytest.mark.parametrize(
    'interruption_value, solution',
    [(0, [4, 7.6, 10.84]), (2, [4.2, 7.98, 11.382])]
)
def test_discounted_accumulated_cost(interruption_value, solution):
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    traj = Trajectory(state_history=[initial_state], plan=[], interruption_probs=[])
    value_cache = {}

    # solution stores the discounted accumulated cost after taking an action
    # from the current state of the trajectory
    for discounted_acc_cost in solution:
        applicable_actions = get_next_actions(traj.state_history[-1], move_actions)
        assert len(applicable_actions) == 1
        assert discounted_accumulated_cost(
            traj,
            applicable_actions[0],
            interruption_value,
            interruption_prob=0.1
        ) == pytest.approx(discounted_acc_cost)

        # update the trajectory
        next_state, _ = get_next_state(traj.state_history[-1], applicable_actions[0])
        traj.level+=1
        traj.state_history.append(next_state)
        traj.plan.append(applicable_actions[0])
        traj.interruption_probs.append(0.1)
        traj.value = discounted_acc_cost
        traj.cost = discounted_acc_cost


def test_h():
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    goal = Fluent("at robot1 living_room") & Fluent("free robot1")

    traj = Trajectory(state_history=[initial_state], plan=[], interruption_probs=[])
    applicable_actions = get_next_actions(initial_state, move_actions)
    assert len(applicable_actions) == 1
    action = applicable_actions[0]

    # tests for when passed in hueristic_fn is an int
    assert h(traj, action, goal, move_actions, 5, next_interruption_prob=0.1) == 0.9 * 5
    traj.interruption_probs.append(0.1)
    assert h(traj, action, goal, move_actions, 5, next_interruption_prob=0.1) == pytest.approx(0.81 * 5)
    traj.interruption_probs.append(0.1)
    assert h(traj, action, goal, move_actions, 5, next_interruption_prob=0.1) == pytest.approx(0.729 * 5)


@pytest.mark.parametrize("heuristic_fn", [0, 5])
def test_construct_trajectory(heuristic_fn):
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    goal = Fluent("at robot1 living_room") & Fluent("free robot1")

    applicable_actions = get_next_actions(initial_state, move_actions)
    assert len(applicable_actions) == 1
    action = applicable_actions[0]
    new_state, next_interruption_prob = get_next_state(initial_state, action, 0.1)

    traj = Trajectory(state_history=[initial_state], plan=[], interruption_probs=[])
    new_traj = traj.create_child(
        goal,
        move_actions,
        action,
        0,
        next_interruption_prob,
        heuristic_fn
    )

    assert new_traj.level == 1
    assert new_traj.cost == 4
    assert new_traj.value == new_traj.cost + heuristic_fn * 0.9
    assert len(new_traj.state_history) == 2
    assert new_traj.state_history == [initial_state, new_state]
    assert len(new_traj.plan) == 1
    assert new_traj.plan == [action]
    assert len(new_traj.interruption_probs) == 1
    assert new_traj.interruption_probs == [0.1]


@pytest.mark.parametrize("heuristic_fn", [0, 5, det_ff_heuristic])
def test_astart_search_noint(heuristic_fn):
    # setup
    move_op = construct_move_operator(4)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
    }

    move_actions = move_op.instantiate(objects_by_type)
    assert len(move_actions) == 2

    initial_state = State(
        time=0,
        fluents={Fluent("at robot1 kitchen"), Fluent("free robot1")}
    )

    goal = Fluent("at robot1 living_room") & Fluent("free robot1")

    # testing with no interrupting tasks
    plan, plan_cost = astar_search(initial_state, goal, move_actions, None, heuristic_fn)
    assert len(plan) == 1
    assert plan[0].name == "move robot1 kitchen living_room"
    assert plan_cost == 4

    # setup test scenario 2
    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
        "object": {"water_bottle"}
    }
    pick_op = construct_pick_operator(3)
    place_op = construct_place_operator(2)

    # instantiate actions
    pick_actions = pick_op.instantiate(objects_by_type)
    move_actions = move_op.instantiate(objects_by_type)
    place_actions = place_op.instantiate(objects_by_type)
    all_actions = pick_actions + move_actions + place_actions

    initial_state = State(
        time=0,
        fluents={
            Fluent("at robot1 kitchen"), Fluent("free robot1"),
            Fluent("at water_bottle kitchen"), ~Fluent("hand-full robot1")
        }
    )

    goal = (
        Fluent("at robot1 living_room") &
        Fluent("free robot1") &
        Fluent("at water_bottle living_room")
    )

    plan, plan_cost = astar_search(initial_state, goal, all_actions, None, heuristic_fn)
    assert len(plan) == 3
    plan_with_names = [action.name for action in plan]
    # print(plan_with_names)
    assert plan_with_names == [
        "pick robot1 kitchen water_bottle",
        "move robot1 kitchen living_room",
        "place robot1 living_room water_bottle"
    ]
    assert plan_cost == pytest.approx(3 + 4 * 0.9 + 2 * 0.81)


@pytest.mark.parametrize("task_distribution", [
    ([Fluent("at robot1 living_room") & Fluent("free robot1")], [1]),
    (
        [
            Fluent("at robot1 living_room") & Fluent("free robot1"),
            Fluent("at robot1 living_room") &
            Fluent("free robot1") &
            Fluent("at water_bottle living_room")
        ], [0.5, 0.5]
    )
])
def test_compute_interruption_value(task_distribution):
    objects_by_type = {
        "robot": {"robot1"},
        "location": {"kitchen", "living_room"},
        "object": {"water_bottle"}
    }
    move_op = construct_move_operator(4)
    pick_op = construct_pick_operator(3)
    place_op = construct_place_operator(2)

    # instantiate actions
    pick_actions = pick_op.instantiate(objects_by_type)
    move_actions = move_op.instantiate(objects_by_type)
    place_actions = place_op.instantiate(objects_by_type)
    all_actions = pick_actions + move_actions + place_actions

    initial_state = State(
        time=0,
        fluents={
            Fluent("at robot1 kitchen"),
            Fluent("free robot1"),
            Fluent("at water_bottle kitchen"),
            ~Fluent("hand-full robot1")
        }
    )
    expected_value = compute_interruption_value(initial_state, all_actions, task_distribution)

    if len(task_distribution[0]) == 1:
        assert expected_value == 4
    else:
        assert expected_value == pytest.approx(0.5*4 + 0.5*(3 + 4 * 0.9 + 2 * 0.81))
