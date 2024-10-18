from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.gp_mpc.controller import GPMPC
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.mission.mission import MissionPlanner
from smallsat_sim.envs.astrobee.cfg.config import randomize_initial_state

import random

def randomize_perturbations(env):
    """
    Randomize what perturbations are applied to the system.
    Ensure that each thruster is perturbed only once by different perturbation scenarios.
    """
    thruster_count = env.model_cfg.Thrusters.n_thrusters  # Assuming the environment has a defined number of thrusters
    thrusters_perturbed = set()  # To track thrusters that have already been perturbed
    perturbation_log = []  # To log the summary of each perturbation applied

    # Define possible perturbation scenarios with the number of thrusters they affect
    perturbation_scenarios = [
        {
            "scenario": "stuck_off_two_thrusters_same_face",
            "probability": 0.35,
            "num_thrusters": 2,
            "action": env.perturbations.perturbations[0].stuck_off_thruster,
        },
        {
            "scenario": "stuck_off_truster",
            "probability": 0.3,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[0].stuck_off_thruster,
        },
        {
            "scenario": "faulty_valve1",
            "probability": 0.3,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[2].register_perturbation,
        },
        {
            "scenario": "faulty_valve2",
            "probability": 0.3,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[2].register_perturbation,
        },
        {
            "scenario": "faulty_valve3",
            "probability": 0.3,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[2].register_perturbation,
        },
        {
            "scenario": "saturated_thrust1",
            "probability": 0.2,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[3].register_perturbation,
        },
        {
            "scenario": "saturated_thrust2",
            "probability": 0.2,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[3].register_perturbation,
        },
        {
            "scenario": "saturated_thrust3",
            "probability": 0.2,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[3].register_perturbation,
        },
        {
            "scenario": "thrust_instability1",
            "probability": 0.05,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[4].register_perturbation,
        },
        {
            "scenario": "thrust_instability2",
            "probability": 0.05,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[4].register_perturbation,
        },
        {
            "scenario": "thrust_instability3",
            "probability": 0.05,
            "num_thrusters": 1,
            "action": env.perturbations.perturbations[4].register_perturbation,
        },
    ]

    # Pairs of thrusters for stuck_off_two_thrusters_same_face scenario
    thruster_pairs = [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11]]  # Define the pairs for the same face

    # Loop through the scenarios
    for perturbation in perturbation_scenarios:
        if random.random() < perturbation['probability']:
            # Determine how many thrusters can still be perturbed
            available_thrusters = [i for i in range(thruster_count) if i not in thrusters_perturbed]
            num_thrusters_to_perturb = min(perturbation['num_thrusters'], len(available_thrusters))

            # If there are enough thrusters available, apply the perturbation
            if num_thrusters_to_perturb > 0:
                if perturbation['scenario'] == 'stuck_off_two_thrusters_same_face':
                    # Randomly select one thruster pair
                    random.shuffle(thruster_pairs)
                    for pair in thruster_pairs:
                        # Check if both thrusters in the pair are available
                        if pair[0] in available_thrusters and pair[1] in available_thrusters:
                            start_time = random.uniform(0, 40)
                            # Apply the perturbation to both thrusters in the pair
                            perturbation['action'](pair[0], start_time)
                            perturbation['action'](pair[1], start_time)
                            # Log the perturbation
                            perturbation_log.append((pair[0], start_time, "stuck_off"))
                            perturbation_log.append((pair[1], start_time, "stuck_off"))
                            # Mark both thrusters as perturbed
                            thrusters_perturbed.add(pair[0])
                            thrusters_perturbed.add(pair[1])
                            break  # Apply perturbation to only one pair and exit

                else:
                    # Randomly sample thrusters from available thrusters for other scenarios
                    selected_thrusters = random.sample(available_thrusters, num_thrusters_to_perturb)

                    # Sample a random starting time
                    start_time = random.uniform(0, 40)
                
                    # Apply the perturbation to each selected thruster
                    for thruster in selected_thrusters:
                        if perturbation['scenario'] == 'stuck_off_truster':
                            perturbation['action'](thruster, start_time)  # Example of stuck_off_thruster action
                            perturbation_log.append((thruster, start_time, "stuck_off"))
                        elif "faulty_valve" in perturbation['scenario']:
                            min_valve_thresh = 0.2 if thruster < 4 else 0.1
                            max_valve_thresh = 0.6 if thruster < 4 else 0.3
                            min_valve = random.uniform(0, min_valve_thresh)
                            max_valve = random.uniform(0.85*max_valve_thresh, max_valve_thresh)

                            perturbation['action'](thruster, start_time, min_valve, max_valve)
                            perturbation_log.append((thruster, start_time, f"faulty_valve ({min_valve}, {max_valve})"))
                        elif "saturated_thrust" in perturbation['scenario']:
                            perturbation['action'](thruster, start_time, 0.0)
                            perturbation_log.append((thruster, start_time, "saturated_thrust"))
                        elif "thrust_instability" in perturbation['scenario']:
                            perturbation['action'](thruster, start_time, 0.0)
                            perturbation_log.append((thruster, start_time, "thrust_instability"))

                        # Mark thruster as perturbed
                        thrusters_perturbed.add(thruster)

    # Print summary of perturbations applied
    print("Perturbation Summary:")
    for thruster, time, p_type in perturbation_log:
        print(f"Thruster {thruster} perturbed at time {time:.2f} with type {p_type}")


# Get arguments for script execution
args = get_args()

# Number of MC runs
num_MC = 10

# Create environment
env = AstrobeeEnv(args=args)
env.reset_perturbations()
randomize_perturbations(env)

# Create planner
planner = MissionPlanner(env)

# Create controller
ctrl = GPMPC(env, planner)


env.env_cfg.sim.max_sim_time = 50

# Simulation loop
for i in range(num_MC):
    if i > 0:
        env.env_cfg.sim.max_sim_time = 120
    else:
        env.env_cfg.sim.max_sim_time = 2
        
    while env.data.time <= env.env_cfg.sim.max_sim_time:
        sim_time = env.data.time

        if True:
            # Calculate control action (open-loop)
            ctrl_input = ctrl.get_control_input(env)

            # Advance simulation
            env.step(input=ctrl_input)

    # Create simulation video if desired
    env.get_sim_rendering(env.env_name)

    # Save log if logging is enabled
    if args.log:
        env.logger.save_log()

    # Increment MC run_id
    env.run_id += 1

    # Sample new initial conditions
    # Returns 3D position and Euler attitude
    pos, att = randomize_initial_state()

    # Reset sim state
    env.reset(pos=pos, att=att)

    # Randomize perturbations
    env.reset_perturbations()
    randomize_perturbations(env)

    # Reinitialze controller
    planner = MissionPlanner(env)
    ctrl = GPMPC(env, planner)

print("Simulation complete.")
