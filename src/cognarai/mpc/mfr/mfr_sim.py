"""Launch Isaac Sim Simulator first."""

import argparse
import zerorpc
import time
import random

from mfr_common import DEFAULT_MFR_TASK_NAME

# IsaacApp Launcher
# -> Must be always created first before importing Omniverse/Issac-related below
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="MFR Isaac Sim Frontend")
parser.add_argument("--task", type=str, default=DEFAULT_MFR_TASK_NAME, help="Name of the task.")
parser.add_argument("--visualize_rollouts", type=int, default=0, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
app_launcher = AppLauncher(args)

"""REST ALL ISAAC & TORCH IMPORTS BELOW"""
# torch
import torch

# isaacsim
from isaacsim.core.version import get_version
print("ISAAC SIM", get_version())
from isaacsim.util.debug_draw import _debug_draw
drawer = _debug_draw.acquire_debug_draw_interface()

# Cognarai
from cognarai.isaac_app import IsaacApp
from cognarai.mppi.utils.transport import bytes_to_torch, torch_to_bytes
from cognarai.mpc.mfr.allegro_env import get_task_config, get_env

def create_rpc_client():
    client = zerorpc.Client()
    client.connect("tcp://127.0.0.1:4242")
    print("MFR server found!")
    return client

def draw_rollouts(rollouts):
    # rollouts: [horizon_length, num_instances, num_bodies, 3]
    drawer.clear_lines()

    # segment: [body[i], body[i+1]], where i: [:-1]
    start_points = rollouts[:, :, :-1, :].reshape(-1, 3).cpu().numpy()
    end_points = rollouts[:, :, 1:, :].reshape(-1, 3).cpu().numpy()
    N = start_points.shape[0]
    colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(N)]
    drawer.draw_lines(start_points, end_points, colors, [1] * N)

def main():
    # Task Config
    task_config = get_task_config(args.task)

    # Planner
    env = get_env(args.task, task_config)
    app = IsaacApp(app_launcher, sim_cfg=env.cfg.sim, args=args)
    app._interactive_scene = env.scene

    # reset environment
    env.reset()

    # create rpc client
    rpc_client = create_rpc_client()

    # simulate environment
    while app.is_running:
        t = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # make a remote call to server to compute actions
            actions = bytes_to_torch(rpc_client.plan_remotely(
                torch_to_bytes(env.get_dof_state()),
                torch_to_bytes(env.get_root_state())
            ))

            # apply actions
            env.step(actions)

            # timekeeping
            print_fps = False
            actual_dt = time.time() - t # time since last step
            sim_dt = env.cfg.sim.dt
            rt = sim_dt / actual_dt
            if rt > 1.0:
                # sleep until [sim_dt] to make synchronous with planner server
                time.sleep(sim_dt - actual_dt)
                if print_fps:
                    # recalculate [actual_dt], only to print FPS
                    actual_dt = time.time() - t
                    rt = sim_dt / actual_dt
            if print_fps:
                print(f"FPS: {1/actual_dt}, RT={rt}")

    # close the simulator
    env.close()
    app.close()


if __name__ == "__main__":
    # run the main function
    main()
