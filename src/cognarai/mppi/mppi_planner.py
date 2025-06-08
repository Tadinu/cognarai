from typing import Callable, Optional
from enum import Enum
import zerorpc
import argparse
import hydra
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="MPPI Isaac Planner")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""REST ALL ISAAC & TORCH IMPORTS BELOW"""
# torch
import torch
torch.set_printoptions(precision=2, sci_mode=False)

# cognarai
from cognarai.mppi.utils.transport import bytes_to_torch, torch_to_bytes
from cognarai.mppi.utils.config_store import MPPITaskBaseCfg, MPPITaskCfg
from cognarai.mppi.mppi_torch import MPPITorch, KMPPITorch
from cognarai.mppi.mppi_kernel import KMPPI, RBFKernel, LinearDeltaDynamics, ScaledLinearDynamics
from cognarai.mppi.mppi_env import MPPIEnv, TASK_NAME, TASK_NAME_ALLEGRO_INHAND
import cognarai.mppi.icem as icem

class PlannerType(Enum):
    MPPI_TORCH = 1
    MPPI_KERNEL = 2
    ICEM = 3

class OmniMPPIPlanner(object):
    """
    Omiverse MPPIPlanner and implements the required functions:
        dynamics, reward, and terminal_cost
    """

    def __init__(self, cfg: MPPITaskCfg, prior: Optional[Callable] = None,
                 planner_type: Optional[PlannerType] = PlannerType.MPPI_TORCH):
        self.cfg = cfg
        self.done = False
        self.planner_type = planner_type

        device = cfg.mppi.device
        dtype = torch.float32 #NOTE: torch.double is not supported by Isaac envs
        self.env = MPPIEnv(
            cfg.env,
            entity_names=cfg.entity_names,
            init_positions=cfg.initial_actor_positions,
            num_envs=cfg.mppi.num_samples,
            device=device,
        )

        if prior:
            self.prior = lambda state, t: prior.compute_command(self.env)
        else:
            self.prior = None

        #print(cfg.mppi)
        using_icem = self.planner_type == PlannerType.ICEM
        using_mppi_torch = not using_icem
        self.mppi = KMPPITorch(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
            dtype=dtype,
        ) if using_mppi_torch else None
        shared_params = {
            "num_samples": cfg.mppi.num_samples,
            "horizon": cfg.mppi.horizon,
            "noise_mu": torch.tensor(cfg.mppi.noise_mu, dtype=dtype, device=device) if cfg.mppi.noise_mu else None,
            "noise_sigma": torch.tensor(cfg.mppi.noise_sigma, dtype=dtype, device=device),
            "u_min": torch.tensor(cfg.mppi.u_min, dtype=dtype, device=device),
            "u_max": torch.tensor(cfg.mppi.u_max, dtype=dtype, device=device),
            "terminal_state_cost": self.terminal_cost,
            "lambda_": cfg.mppi.lambda_,
            "device": device,
            "dtype": dtype
        }
        # self.dynamics = ScaledLinearDynamics(self.running_cost, B), self.running_cost, 2,
        self.mppi_kernel = KMPPI(**shared_params,
                                 dynamics=self.dynamics,
                                 kernel=RBFKernel(sigma=2),
                                 running_cost=self.running_cost,
                                 nx=cfg.nx,
                                 num_support_pts=5) if using_mppi_torch and not self.mppi else None
        print("USE MPPI TORCH", using_mppi_torch, "NUM SAMPLES", cfg.mppi.num_samples,
              "HORIZON", cfg.mppi.horizon)

        nu = len(cfg.mppi.u_min)
        self.icem = icem.ICEM(self.dynamics, icem.accumulate_running_cost(self.running_cost, self.terminal_cost), nx=cfg.nx,
                              nu=nu, sigma=torch.ones(nu, dtype=dtype, device=device),
                              warmup_iters=100, online_iters=100,
                              num_samples=cfg.mppi.num_samples, num_elites=10,
                              horizon=cfg.mppi.horizon, device=device)
        # Note: place_holder variable to pass to mppi so it doesn't complain, while the real state is actually the isaaclab simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))
        #print("MPPI inited", OmegaConf.to_yaml(self.mppi.cfg))

    def dynamics(self, _, u, t: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaaclab the state is already saved in the simulator itself, so we ignore it.
        # Note: t is an unused step dependent dynamics variable
        self.env.step(u)
        return self.state_place_holder, u

    def running_cost(self, state:  Optional[torch.Tensor] = None, u: Optional[torch.Tensor] = None):
        return self.env.cost

    def terminal_cost(self, states, actions):
        terminal_scale = 10 if self.planner_type == PlannerType.ICEM else 100
        return terminal_scale * self.running_cost(states[..., -1, :] if states is not None else None,
                                                  actions[..., -1, :] if actions is not None else None)

    def reset_rollout_sim(self, dof_state_data: bytes, root_state_data: bytes):
        self.env.visual_link_buffer = []
        dof_state = bytes_to_torch(dof_state_data)
        size = self.env.get_robot_joint_pos_shape()[0]
        #print("reset_rollout_sim:")
        #print("joint_pos_size", self.env.get_robot_joint_pos_shape())
        #print("dof_state", dof_state.shape)
        #print("root_state", bytes_to_torch(root_state_data).shape)
        joint_pos = dof_state[:size,:]
        joint_vel = dof_state[size:,:]
        self.env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.env.robot.write_root_state_to_sim(bytes_to_torch(root_state_data))

    def compute_action_tensor(self, dof_state_data: bytes, root_state_data: bytes):
        """
        Remote client procedural call
        """
        self.reset_rollout_sim(dof_state_data, root_state_data)
        action = self.mppi.command(self.state_place_holder) if self.mppi else \
            self.mppi_kernel.command(self.state_place_holder) if self.mppi_kernel else (
                self.icem.command(self.state_place_holder))
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return torch_to_bytes(action)

    def add_to_env(self, env_cfg_additions):
        self.env.add_to_envs(env_cfg_additions)

    def get_rollouts(self):
        return torch_to_bytes(torch.stack(self.env.visual_link_buffer))

@hydra.main(version_base=None, config_path="cfg/task", config_name=TASK_NAME)
def main(task_base_cfg: MPPITaskBaseCfg):
    #print(f"{TASK_NAME} base config", OmegaConf.to_yaml(task_base_cfg))
    task_cfg = MPPITaskCfg.from_parent(task_base_cfg)
    planner = zerorpc.Server(OmniMPPIPlanner(task_cfg, planner_type=PlannerType.ICEM))
    planner.bind("tcp://0.0.0.0:4242")
    print("Planner server running..")
    planner.run()


if __name__ == "__main__":
    main()


