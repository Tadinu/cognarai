from dataclasses import dataclass, field
import torch
import numpy as np
from enum import Enum
from typing import List, Optional, Any

# IsaacLab
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ViewerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv
from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import FrankaCabinetEnv, FrankaCabinetEnvCfg

# Cognarai
from cognarai.mppi.utils.isaac_utils import load_entity_cfgs

TASK_NAME_ALLEGRO_INHAND = "allegro_inhand"
TASK_NAME_FRANKA_CABINET = "franka_cabinet"
TASK_NAME = TASK_NAME_ALLEGRO_INHAND

TaskEnvCfgDict = {
    TASK_NAME_FRANKA_CABINET: FrankaCabinetEnvCfg,
    TASK_NAME_ALLEGRO_INHAND: AllegroHandEnvCfg
}

TaskEnvDict = {
    TASK_NAME_FRANKA_CABINET: FrankaCabinetEnv,
    TASK_NAME_ALLEGRO_INHAND: InHandManipulationEnv
}

@configclass
class MPPIBaseEnvCfg(TaskEnvCfgDict[TASK_NAME]):
    obs_type = 'openai'
    def __post_init__(self):
        self.robot_cfg: ArticulationCfg = self.robot if hasattr(self, "robot") else self.robot_cfg

class MPPIBaseEnv(TaskEnvDict[TASK_NAME]):
    pass

@configclass
class MPPIEnvCfg(MPPIBaseEnvCfg):
    spacing: float = 6.0
    #state_space = 0

    # robot
    extra_robot_cfgs: list[ArticulationCfg] = []

    # viewer
    viewer = ViewerCfg(
        eye=[7.5, 7.5, 7.5]
    )

    # env
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_client_threads: int = 0
    num_obstacles: int = 0


class SupportedEntityTypes(Enum):
    Axis = 1
    Robot = 2
    Sphere = 3
    Box = 4


@dataclass
class EntityCfg:
    type: SupportedEntityTypes
    name: str
    dof_mode: str = "velocity"
    init_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    init_ori: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    size: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0  # kg
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    fixed: bool = False
    collision: bool = True
    friction: float = 1.0
    handle: Optional[int] = None
    flip_visual: bool = False
    urdf_file: str = None
    visualize_link: str = None
    gravity: bool = True
    differential_drive: bool = False
    init_joint_pose: List[float] = None
    wheel_radius: Optional[float] = None
    wheel_base: Optional[float] = None
    wheel_count: Optional[float] = None
    left_wheel_joints: Optional[List[str]] = None
    right_wheel_joints: Optional[List[str]] = None
    caster_links: Optional[List[str]] = None
    noise_sigma_size: Optional[List[float]] = None
    noise_percentage_mass: float = 0.0
    noise_percentage_friction: float = 0.0


class MPPIEnv(MPPIBaseEnv):
    def __init__(
        self,
        cfg: MPPIEnvCfg,
        entity_names: Optional[List[str]] = None,
        init_positions: Optional[List[List[float]]] = None,
        num_envs: int = 1,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        # NOTE: A bit hacky here, just to avoid duplicate cfgs from outside in Planner server & Sim client
        cfg.scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=cfg.spacing, replicate_physics=True)
        cfg.num_envs = num_envs
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg: MPPIEnvCfg = cfg
        self.sim_cfg = cfg.sim
        self.scene_cfg = cfg.scene
        self.scene_cfg.num_envs = num_envs
        self.visual_link_buffer = []
        if entity_names:
            assert len(entity_names) == len(set(entity_names)), f"{entity_names} contains duplicate names"
        self.entity_cfgs = load_entity_cfgs(entity_names) if entity_names else []

        # TODO: check for initial position collisions of entity_names
        if init_positions is not None:
            robot_cfgs = [a for a in self.entity_cfgs if a.type == "robot"]
            if robot_cfgs:
                assert len(robot_cfgs) == len(init_positions)
                for init_pos, _ in zip(init_positions, robot_cfgs):
                    robot_cfg = ArticulationCfg()
                    robot_cfg.init_state.pos = init_pos
                    self.cfg.extra_robot_cfgs.append(robot_cfg)
            else:
                self.cfg.robot_cfg.init_state.pos = tuple(init_positions[0])

    def _setup_scene(self):
        super()._setup_scene()
        self.extra_robots = [Articulation(cfg) for cfg in self.cfg.extra_robot_cfgs]

    @property
    def robot(self):
        return self._robot

    def get_root_state(self) -> torch.Tensor:
        return self.robot.data.root_state_w.clone()

    def get_dof_state(self) -> torch.Tensor:
        return torch.concatenate([self.robot.data.joint_pos.clone(), self.robot.data.joint_vel.clone()])

    def get_robot_joint_pos_shape(self):
        return self.robot.data.joint_pos.shape

    def step(self, action: torch.Tensor):
        #print("ACTION-------------------", action, action.shape)
        assert action.dim() == 2, f"action shape: {action.shape}" # (1, actions_num)
        res = super().step(action)
        self.visual_link_buffer.append(self.robot.data.body_link_pos_w.clone())
        return res
