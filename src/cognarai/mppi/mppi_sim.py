"""Launch Isaac Sim Simulator first."""

import argparse
import zerorpc
import time
import random

from isaaclab.app import AppLauncher
"""NOTE: NO IMPORT HERE JUST YET"""

# add argparse arguments
parser = argparse.ArgumentParser(description="MPPI Isaac Sim Frontend")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--visualize_rollouts", type=int, default=0, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""REST ALL ISAAC & TORCH IMPORTS BELOW"""
# torch
import torch

# isaacsim
from isaacsim.core.version import get_version
print("ISAAC SIM", get_version())
from isaacsim.util.debug_draw import _debug_draw
drawer = _debug_draw.acquire_debug_draw_interface()

# cognarai
from cognarai.mppi.mppi_env import MPPIEnvCfg, MPPIEnv, TASK_NAME
from cognarai.mppi.utils.transport import torch_to_bytes, bytes_to_torch
from cognarai.mppi.utils.config_store import load_mppi_config
if args_cli.task is None:
    args_cli.task = TASK_NAME
# PLACEHOLDER: Extension template (do not remove this comment)

def create_rpc_client():
    client = zerorpc.Client()
    client.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")
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
    task_name = args_cli.task
    visualize_rollouts = args_cli.visualize_rollouts

    # load task's mppi config
    mppi_cfg = load_mppi_config(task_name)

    # create environment configuration
    env_cfg = MPPIEnvCfg()
    sim_cfg = env_cfg.sim
    sim_cfg.device = args_cli.device
    sim_cfg.num_envs = 1
    sim_cfg.use_fabric = not args_cli.disable_fabric
    env_cfg.viewer.eye = [1.0, 1.0, 2.0]
    env_cfg.viewer.lookat = [0.0, 0.0, 1.0]

    # create environment
    env = MPPIEnv(cfg=env_cfg, num_envs=sim_cfg.num_envs)
    print(f"[INFO]: Env observation space: {env.observation_space}")
    print(f"[INFO]: Env action space: {env.action_space}")
    print(f"[INFO]: Envs num: {env.num_envs}")

    # reset environment
    env.reset()

    # create rpc client
    rpc_client = create_rpc_client()

    # simulate environment
    while simulation_app.is_running():
        t = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # make a remote call to server to compute actions
            actions = bytes_to_torch(
                rpc_client.compute_action_tensor(
                    torch_to_bytes(env.get_dof_state().repeat(mppi_cfg.num_samples, 1)),
                    torch_to_bytes(env.get_root_state().repeat(mppi_cfg.num_samples, 1))
                )
            )

            # apply actions
            env.step(actions)

            # Visualize rollouts
            if visualize_rollouts:
                rollouts = bytes_to_torch(rpc_client.get_rollouts())
                draw_rollouts(rollouts)

            # timekeeping
            print_fps = False
            actual_dt = time.time() - t # time since last step
            sim_dt = env_cfg.sim.dt
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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
