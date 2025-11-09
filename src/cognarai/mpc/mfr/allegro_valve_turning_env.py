from __future__ import annotations
from typing import Tuple, Optional, Sequence
import os
from pathlib import Path
import math
import yaml

# Third-party
import torch

# Isaac Lab
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg

# cognarai
from cognarai.mpc.mfr.allegro_env import AllegroManipEnv, AllegroManipEnvCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = f"{CURRENT_DIR}/config"
MODELS_DIR = f"{CURRENT_DIR}/models"
ALLEGRO_URDF_DIR = f"{MODELS_DIR}/allegro_xela"
VALVE_URDF_DIR = f"{MODELS_DIR}/valve"


# -----------------------------------------------------------------------------
# Task object (valve)
# -----------------------------------------------------------------------------

@configclass
class ValveObjectCfg(RigidObjectCfg):
    """Simple cylinder acting as a valve the hand can grasp/turn."""

    prim_path = "/World/envs/env_.*/Task/Valve"
    spawn = sim_utils.CylinderCfg(
        radius=0.03,  # 3 cm
        height=0.05,  # 5 cm tall
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # initial state is identity by default; we randomize at reset


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

VALVE_INIT_POS = [0.25, 0., 0.31]
VALVE_INIT_QUAT = [0.71, 0.0, 0.0, -0.71]

# -----------------------------------------------------------------------------
# Valve Turning environment config
# -----------------------------------------------------------------------------
"""
num_envs=num_envs,
control_mode='joint_impedance',
viewer=True,
steps_per_action=60,
friction_coefficient=1.0,
device=config['sim_device'],
video_save_path=img_save_dir,
joint_stiffness=config['kp'],
fingers=config['fingers'],
gravity=config['gravity'],
gradual_control=config['gradual_control'],
"""


@configclass
class AllegroValveTurningCfg(AllegroManipEnvCfg):
    # simulation / scene
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.5, replicate_physics=True)

    # valve (placeholder uses an instanceable USD box asset; replace if you have USD export of the URDF)
    # NOTE: Valve is an Articulation, not a RigidBody
    object_urdf_path: str = f"{VALVE_URDF_DIR}/valve_cross.urdf"
    valve_cfg: ArticulationCfg = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{Path(object_urdf_path).stem}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=object_urdf_path,
            fix_base=True,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            activate_contact_sensors=True,
            merge_fixed_joints=False,
            make_instanceable=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=VALVE_INIT_POS, rot=VALVE_INIT_QUAT),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        }
    )

    # Contact with fingers
    valve_body_name: str = "valve"
    contact_sensor_cfg: ContactSensorCfg = AllegroManipEnvCfg().contact_sensor_cfg.replace(
        prim_path=f"{valve_cfg.prim_path}/{valve_body_name}"
    )

    # action/observation sizes; action -> 16 allegro joint deltas
    action_space: int = 16
    observation_space: int = 64

    # default joint initial pose (derived from allegro.py default_dof_pos)
    default_q: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.object_cfg = self.valve_cfg
        robot_init_state_cfg = self.robot_cfg.init_state
        robot_init_state_cfg.pos = [-0.1, 0.0, 0.30]
        robot_init_state_cfg.rot = [1, 0, 0, 0]
        self.viewer.eye = [0.3, -0.3, 1.1]
        self.viewer.look_at = [0.0, 0.0, 0.31]


# -----------------------------------------------------------------------------
# Environment implementation
# Original source: https://github.com/UM-ARM-Lab/MFR_benchmark
# -----------------------------------------------------------------------------

class AllegroValveTurningEnv(AllegroManipEnv):
    def __init__(self, task_cfg: dict, render_mode: Optional[str] = None, **kwargs):
        super().__init__(task_cfg=task_cfg,
                         cfg=AllegroValveTurningCfg(fingers=task_cfg['fingers']), render_mode=render_mode, **kwargs)

        # target yaw we want to achieve (per-env) — the task: rotate valve to this yaw
        self.target_yaw = torch.zeros((self.scene.num_envs,), device=self.device)

        self.default_dof_pos = torch.cat((torch.tensor([[0.0, 0.8, 0.4, 0.7]]).float(),
                                          torch.tensor([[-0.15, 0.9, 1.0, 0.9]]).float(),
                                          torch.tensor([[0, 0.3, 0.3, 0.6]]).float(),
                                          torch.tensor([[0.7, 1.0, 0.6, 1.05]]).float()),
                                         dim=1).to(self.device)

        # add the screwdriver angle to it
        self.default_dof_pos = torch.cat(
            (self.default_dof_pos, torch.tensor([[0, 0, 0, 0, -0.523599, 0]]).float().to(device=self.device)),
            dim=1)
        self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.reset()

    # ---- Scene building -----------------------------------------------------

    def _setup_scene(self):
        assert isinstance(self.cfg, AllegroValveTurningCfg)
        super()._setup_scene()

        # valve articulation
        self.valve = Articulation(self.cfg.valve_cfg)
        self.object = self.valve
        self.scene.articulations["valve"] = self.valve

    def spawn_entity_from_urdf(self, obj_urdf_path, obj_pos, obj_quat) -> RigidObject:
        cfg = sim_utils.UrdfFileCfg(
            asset_path=obj_urdf_path,
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        )
        prim_path = f"/World/envs/env_.*/{Path(obj_urdf_path).stem}"
        cfg.func(prim_path, cfg, translation=obj_pos, orientation=obj_quat)
        return RigidObject(RigidObjectCfg(prim_path=prim_path))

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()

        assert len(self.actions.shape) == 2
        state = self.get_state()

        # goal cost
        obj_pos = self.object_pos
        obj_ori = self.object_rot

        valve_upright_cost = (obj_ori[:, 0] ** 2) + (obj_ori[:, 2] ** 2)
        reward -= 1000 * valve_upright_cost

        dropp_flag = state[:, -4] < -0.07
        # reward -= 1000 * dropp_flag.to(self.device)
        # dropping cost
        reward -= 1e6 * (dropp_flag * state[:, -4]) ** 2

        # action_cost
        reward -= 50.0 * (torch.norm(self.actions, dim=-1) ** 2)

        # small penalty for high joint velocities
        reward -= 0.01 * torch.linalg.norm(self.hand_dof_vel)
        return reward

    def get_state(self):
        results = super().get_state()
        valve_pos = self.valve.data.root_pos_w - self.scene.env_origins
        angles = euler_xyz_from_quat(self.object.data.root_quat_w)
        valve_rot = torch.tensor([angles[0], angles[1], angles[2]], device=self.device).reshape(1,
                                                                                                len(angles))
        q = []
        for finger in self.finger_names:
            q.append(results[f'{finger}_q'])

        # obj_dof_code = self.task_cfg['obj_dof_code']
        q.append(valve_pos)
        q.append(valve_rot)
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)

        N = env_ids.numel()
        # randomize valve pose, maintaining Z
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos = torch.zeros((N, 3), device=self.device)
        pos[:, 0] = 0.0 + 0.02 * (torch.rand(N, device=self.device) - 0.5)
        pos[:, 1] = 0.0 + 0.02 * (torch.rand(N, device=self.device) - 0.5)
        pos[:, 2] = 0.31
        object_default_state[:, 0:3] = pos

        yaw = (torch.rand(N, device=self.device) - 0.5) * 2 * math.pi
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        quat = torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), sy, cy], dim=-1)
        object_default_state[:, 3:7] = quat

        self.valve.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.valve.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids=env_ids)

        # set per-env target yaw (e.g. random target in [-pi,pi])
        self.target_yaw[env_ids] = (torch.rand(N, device=self.device) - 0.5) * 2 * math.pi
