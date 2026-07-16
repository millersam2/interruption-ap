from railroad.environment.procthor.environment import ProcTHOREnvironment
from railroad.core import Operator
from railroad import operators
from utilities import (
    construct_gripper_pick_operator,
    construct_gripper_place_operator
)

class KitchenProcTHOREnvironment(ProcTHOREnvironment):
    """
    Kitchen ProcTHOR environment with relevant internal operator construction.
    """
    def define_operators(self) -> list[Operator]:
        move_op = operators.construct_move_operator(self.estimate_move_time)
        pick_op = construct_gripper_pick_operator(10.0)
        place_op = construct_gripper_place_operator(10.0)
        # previously extra_cost was set to 100.0
        # no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=0)
        return [pick_op, place_op, move_op]
