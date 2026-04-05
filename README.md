# interruption-ap
Project Repository for Anticipatory Planning for Interruptions

Progress Log

4/3 - 4/5:

Implementation:
TODO - 
- convert basic scenario to use the interruption_ap planner

DONE - 
- Data Class for trajectories:
    attributes:
        accumulated cost to get to state
        level of tree
        state history (list)
        plan
- astar_search
    Inputs: state, goal, actions, num_steps, interruption_task_values, heuristic_fn
    Outputs: trajectory
- interrupt_value: computes the cost of completing interrupting task from a state (want the values to be cached)
    Inputs: state, task
    Outputs: cost
- accumulated cost function g(p)
    Inputs: trajectory (used to access states before and after transition), action, cached value of interrupt
    Outputs: cost
- heuristic function h(p) [used as estimate of q-value of transitioned state under no interruption]
    Inputs: trajectory, action, desired heuristic function
    Outputs: heuristic cost
- value function v(p)

Testing:

TODO -

DONE - 
- get_action_cost
- get_next_state
- check_value_cache
- g
- h
- construct_trajectory
- compute_interruption_value
- astar_search