import pytest
from railroad.operators.core import construct_pick_operator, construct_move_operator
from railroad.core import State, Fluent, get_next_actions
from utilities import get_action_cost, get_next_state

@pytest.mark.parametrize("action_cost", [1, 3, 5])
def test_get_action_cost_pick(action_cost):
    pick_op = construct_pick_operator(action_cost)

    objects_by_type = {
        "robot": {"robot1"},
        "location": {"pantry", "refrigerator"},
        "object": {"turkey", "bread"}
    }

    pick_actions = pick_op.instantiate(objects_by_type)

    for action in pick_actions:
        assert get_action_cost(action) == action_cost

def test_get_next_state():
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

    applicable_actions = get_next_actions(initial_state, move_actions)
    assert len(applicable_actions) == 1
    next_state, interruption_prob = get_next_state(initial_state, applicable_actions[0])
    assert next_state.time == 4
    assert next_state.fluents == {Fluent("at robot1 living_room"), Fluent("free robot1")}
    assert interruption_prob == 0
