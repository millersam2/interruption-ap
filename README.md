# interruption-ap
Project Repository for Anticipatory Planning for Interruptions

Progress Log

4/3 - 4/12:

Implementation:
TODO - 
- implement visited list to help with search time
- closure for heuristic function
- implement interruption_planner class to get the dashboard to work
- basic interruption scenario with debug environment (no interruptions), for probability param tuning
- basic interruption scenario with tuned parameter for actual interruption environment

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
- code changes to increase readability based on code review with Professor Stein:
    - rename g to discounted_accumulated_cost
    - remove metastate as an argument to astar
    - construct_trajectory read from and pass value instead of whole cache
    - construct_trajectory implement as method rather than function
- revamp how interruption probabilities are defined/used
- implemented comparison operators for trajectory class due to error being raised when pushing tuples of (q_value, traj)
    on to the frontier (due to multiple trajectories having the same q_value)


Testing:

TODO -
- add some additional context to tests
- add additional testing for interruption_prob_fn

DONE - 
- get_action_cost
- get_next_state
- check_value_cache
- g
- h
- construct_trajectory
- compute_interruption_value
- astar_search
- split g into two tests using a helper function