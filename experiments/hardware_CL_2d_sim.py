import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.utils.conversions import yaw_from_quaternion
from smallsat_sim.controllers.mpcc.controller_2d import NominalMPCCController2D
from smallsat_sim.envs.smallsat_hardware_2d.env import SmallsatHardware2DEnv
# Different path options
from smallsat_sim.planners.mission.mission_2d_square import MissionPlanner2DSquare
from smallsat_sim.planners.mission.mission_2d_square_small import MissionPlanner2DSquareSmall 
from smallsat_sim.planners.mission.mission_2d_x_line import MissionPlanner2DxLine
from smallsat_sim.planners.mission.mission_2d_y_line import MissionPlanner2DyLine
from smallsat_sim.planners.mission.mission_2d_rot import MissionPlanner2Drot

import csv
import math
from datetime import datetime
import pytz
import os

LOG_DATA = True
MISSION_PLAN = "y_line" # Options: x_line, y_line, square_small, square

print("----------------------------")
print(f"Logging data: {LOG_DATA}")
print(f"Mission plan: {MISSION_PLAN}")
print("----------------------------")
#---------------- UTILS ---------------- 

def quaternion_to_euler(x, y, z, w):
    """
    Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw  # Returns angles in radians

#---------------- LOG DATA ----------------
if LOG_DATA:
    timezone = pytz.timezone('America/Los_Angeles')
    now = datetime.now(timezone)
    date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S %Z%z")   
    os.makedirs(f"logs/sim/{date}", exist_ok=True)  # Creates folder if it doesn't exist
    # Generate filename with date and time
    file_path = os.path.join(f"logs/sim/{date}", f"{current_time}.csv")

    # Column headers
    headers = ["Timestamp", "obs_pos_x", "obs_pos_y", "obs_pos_z", "obs_euler_roll", "obs_euler_pitch", "obs_euler_yaw", "obs_quat_w", "obs_quat_x", "obs_quat_y", "obs_quat_z", "obs_vel_x", "obs_vel_y", "obs_vel_z", "obs_rot_roll", "obs_rot_pitch", "obs_rot_yaw", "ref_x", "ref_y", "ref_att", "control_input", "solve_time", "control_callback_time"]

    # Write headers if file is empty
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if file is empty
            writer.writerow(headers)

# -------------- RUN SIM --------------
# Get arguments for script execution
args = get_args()
print("args: ", args)

# Create environment
env = SmallsatHardware2DEnv(args=args)

# Create planner
if MISSION_PLAN == "square":
    planner = MissionPlanner2DSquare(env)
elif MISSION_PLAN == "square_small":
    planner = MissionPlanner2DSquareSmall(env)
elif MISSION_PLAN == "x_line":
    planner = MissionPlanner2DxLine(env)
elif MISSION_PLAN == "y_line":
    planner = MissionPlanner2DyLine(env)
elif MISSION_PLAN == "rot":
    planner = MissionPlanner2Drot(env)
else:
    print("No mission planner defined. Defaulting to y_line")
    planner = MissionPlanner2DyLine(env)

# Create controller
ctrl = NominalMPCCController2D(env, planner)

# Define start time
start_time = time.time()

# Simulation loop
while env.data.time <= env.env_cfg.sim.max_sim_time:
    real_time = time.time() - start_time

    sim_time = env.data.time

    if True:

        obs = env.get_obs()
        # print("---------------------------")
        # print("Got obs: ")
        # print(f"position: {obs[0], obs[1], obs[2]}")
        q = [obs[3], obs[4], obs[5], obs[6]]
        yaw = yaw_from_quaternion(q)
        # print(f"yaw: {yaw}")
        # print(f"velocities: {obs[7], obs[8], obs[9]}")
        # print(f"yaw rate: {obs[12]}")
        # ref_pt_pos, ref_pt_att = planner.get_reference(obs)
        ref_pt_pos, ref_pt_att = planner._get_reference_intermediate_wp_tracking(obs)
        ref_x = ref_pt_pos[0]
        ref_y = ref_pt_pos[1]
        ref_att = ref_pt_att[2]
        print(f"Reference pos: {ref_x, ref_y, ref_pt_pos[2]}")
        # print(f"Reference att: {ref_pt_att[0], ref_pt_att[1], ref_att, ref_pt_att[3]}")
        ref_yaw = yaw_from_quaternion(ref_pt_att)
        print(f"Reference yaw: {ref_yaw}")
        # e_yaw = (yaw - 0) % (2 * math.pi)  # Wrap to [0, 2π]
        e_yaw = ((yaw - 0)+math.pi) % (2*math.pi) - math.pi # Wrap to [0, 2π]
        # print(f'Yaw: {yaw}')
        # print(f'Yaw error: {e_yaw}')
        # print('--------------------------')

        # Calculate control action (open-loop)
        ctrl_input = ctrl.get_control_input(env)

        if LOG_DATA:
            # obs = [r (3),q (4),v (3), --> in body/inertial frame, omega (3)]
            obs = env.get_obs()
            # print(f"obs: {obs}")
            obs_pos_x = obs[0]
            obs_pos_y = obs[1]
            obs_pos_z = obs[2]
            obs_quat_w = obs[3]
            obs_quat_x = obs[4]
            obs_quat_y = obs[5]
            obs_quat_z = obs[6]
            obs_vel_x = obs[7]
            obs_vel_y = obs[8]
            obs_vel_z = obs[9]
            obs_rot_roll = obs[10]
            obs_rot_pitch = obs[11]
            obs_rot_yaw = obs[12]
            # print(f"obs_pos: {obs_pos_x, obs_pos_y, obs_pos_z}, obs_quat: {obs_quat_w, obs_quat_x, obs_quat_y, obs_quat_z}, obs_vel: {obs_vel_x, obs_vel_y, obs_vel_z}, obs_rot: {obs_rot_roll, obs_rot_pitch, obs_rot_yaw}")
            roll, pitch, yaw = quaternion_to_euler(obs_quat_x, obs_quat_y, obs_quat_z, obs_quat_w)
            obs_euler_roll = roll
            obs_euler_pitch = pitch
            obs_euler_yaw = yaw

            ref_pt_pos, ref_pt_att = planner.get_reference(obs)
            ref_x = ref_pt_pos[0]
            ref_y = ref_pt_pos[1]
            ref_att_yaw = yaw_from_quaternion(ref_pt_att)
            ref_att = ref_att_yaw
            solve_time = ctrl.solve_time
            ctrl_loop_duration = ctrl.ctrl_input_callback_time
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")       
                # ["Timestamp", "obs_pos", "obs_euler", "obs_quat", "obs_vel", "obs_rot", "control_input"]     
                writer.writerow([timestamp, obs_pos_x, obs_pos_y, obs_pos_z, obs_euler_roll, obs_euler_pitch, obs_euler_yaw, obs_quat_w, obs_quat_x, obs_quat_y, obs_quat_z, obs_vel_x, obs_vel_y, obs_vel_z, obs_rot_roll, obs_rot_pitch, obs_rot_yaw, ref_x, ref_y, ref_att, ctrl_input, solve_time, ctrl_loop_duration])

        # for i in range(8):
        #     print(f"Control input Th{i+1}: {ctrl_input[i]}")
        # print("---------------------------------------")

        # Advance simulation
        env.step(input=ctrl_input)

print("Out of while loop")
# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()