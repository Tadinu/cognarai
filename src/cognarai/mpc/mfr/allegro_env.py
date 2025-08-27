"""
Allegro Hand manipulation environments
-----------------------------------------------------------------
This file ports a custom Isaac Gym Allegro-hand task to the Isaac Lab
"direct RL" workflow. It targets Isaac Sim 5.0 and the latest
Isaac Lab API (see docs linked in the chat response for references).
- Uses `DirectRLEnv` / `DirectRLEnvCfg` and the InteractiveScene.
- Spawns an Allegro hand articulation and task objects
- Supports joint-position or joint-velocity control through PD actuators.
- Vectorized over N environments; designed for RL (Gymnasium-like API).
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence
import pathlib
import math

# Third-party
import numpy as np
import torch
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf

# Isaac Lab
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv
from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg

MODELS_DIR = f"{pathlib.Path(__file__).parent.resolve()}/models"
ALLEGRO_URDF_DIR = f"{MODELS_DIR}/allegro_xela"

# -----------------------------------------------------------------------------
# Task objects (valve / screwdriver)
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


@configclass
class ScrewdriverObjectCfg(RigidObjectCfg):
    """Screwdriver asset (replace USD path to your tool)."""

    prim_path = "/World/envs/env_.*/Task/Screwdriver"
    spawn = sim_utils.UsdFileCfg(
        usd_path="/Nucleus/Assets/Tools/screwdriver.usd",  # TODO: set your path
        scale=(1.0, 1.0, 1.0),
    )


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

@configclass
class AllegroManipEnvCfg(AllegroHandEnvCfg):
    # Simulation & runtime
    decimation = 2                    # control at sim_dt * decimation
    episode_length_s = 10.0           # per-episode horizon (seconds)

    # Spaces (update if you change obs/action later)
    action_space = 16                 # Allegro has 16 actuated joints
    observation_space = 64            # joint pos/vel + object pose/vel, etc.
    state_space = 0                   # no privileged state by default

    # Allegro hand with Xela sensors
    actuated_joint_names: list[str] = [
        'allegro_hand_hitosashi_finger_finger_joint_0',
        'allegro_hand_naka_finger_finger_joint_4',
        #'allegro_hand_kusuri_finger_finger_joint_8',
        'allegro_hand_oya_finger_joint_12',
        'allegro_hand_hitosashi_finger_finger_joint_1',
        'allegro_hand_hitosashi_finger_finger_joint_2',
        'allegro_hand_hitosashi_finger_finger_joint_3',
        'allegro_hand_naka_finger_finger_joint_5',
        'allegro_hand_naka_finger_finger_joint_6',
        'allegro_hand_naka_finger_finger_joint_7',
        #'allegro_hand_kusuri_finger_finger_joint_9',
        #'allegro_hand_kusuri_finger_finger_joint_10',
        #'allegro_hand_kusuri_finger_finger_joint_11',
        'allegro_hand_oya_finger_joint_13',
        'allegro_hand_oya_finger_joint_14',
        'allegro_hand_oya_finger_joint_15',
    ]
    fingers: list[str] = [] #['index', 'middle', 'thumb'] # 'ring'
    fingertip_body_names: list[str] = [
        "hitosashi_ee", # Index~
        "naka_ee", # Middle~
        #"kusuri_ee", # Ring~
        "oya_ee",# Thumb~
    ]

    # Physics stepping
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",  # GPU mode; PhysX GPU and pipeline are chosen automatically
        dt=1. / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2 ** 23,
            # other GPU buffer sizes
        )
    )

    # Scene cloning (vectorized envs)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=0.5,
        replicate_physics=True,
    )

    # robot
    arm_type: str = None
    hand_urdf_path: str = f"{ALLEGRO_URDF_DIR}/victor_allegro.urdf" if arm_type == 'robot' \
        else f"{ALLEGRO_URDF_DIR}/allegro_hand_right_floating_3d.urdf" if arm_type == 'floating_3d' \
        else f"{ALLEGRO_URDF_DIR}/allegro_hand_right_floating_6d.urdf" if arm_type == 'floating_6d' \
        else f"{ALLEGRO_URDF_DIR}/allegro_hand_right.urdf"
    robot_init_pose: list[list[float]] = [[-0.1, -0.025, 0.30],[0.258819, 0, 0, 0.9659258]]
    robot_cfg: ArticulationCfg = (ALLEGRO_HAND_CFG.
        replace(prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UrdfFileCfg(
                    fix_base=True,
                    merge_fixed_joints=False,
                    make_instanceable=True,
                    asset_path=hand_urdf_path,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
                    )),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=robot_init_pose[0],
                        rot=robot_init_pose[1],
                        joint_pos={"^(?!allegro_hand_hitosashi_finger_finger_joint_0).*": 0.28, "allegro_hand_hitosashi_finger_finger_joint_0": 0.28},
    )))

    # Default joint targets used at reset
    default_q: float = 0.5

    # Object
    # Choose which object to spawn; set only one True
    use_valve: bool = False
    use_screwdriver: bool = False

    valve_cfg: RigidObjectCfg = ValveObjectCfg()
    screwdriver_cfg: RigidObjectCfg = ScrewdriverObjectCfg()
    obj_pos: list[float] = [0, 0, 0.31]

# -----------------------------------------------------------------------------
# Environment (Direct RL Workflow)
# -----------------------------------------------------------------------------

class AllegroManipEnv(InHandManipulationEnv):
    """Allegro-hand manipulation task in Isaac Lab's direct RL workflow.

    Observations
    -----------
    Concatenated vector with:
    - Allegro joint positions/velocities (normalized)
    - Object pose (p, q) and velocities
    - (Optional) fingertip poses

    Actions
    -------
    - By default: joint position deltas in [-1, 1], scaled to per-joint range.
      Switch to velocity control by changing `_apply_action`.
    """
    def __init__(self, cfg: AllegroManipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # World, obj transf
        self.world_trans = tf.Transform3d(pos=cfg.robot_init_pose[0], rot=cfg.robot_init_pose[1], device=self.device)
        self.object_init_pos = torch.tensor(cfg.obj_pos, device=self.device)

        # Others
        self.finger_names = cfg.fingers
        self.finger_to_joint_index = {
            'index': [0, 1, 2, 3],
            'middle': [8, 9, 10, 11],
            'ring': [4, 5, 6, 7],
            'thumb': [12, 13, 14, 15]
        }

        body_names = self.hand.data.body_names
        self.finger_ee_index = {
            finger: [body_names.index(fingertip_name) for fingertip_name in self.cfg.fingertip_body_names]
            for finger in self.finger_names
        }

        self._rb_states, self.rb_states = None, None
        self._actor_rb_states, self.actor_rb_states = None, None
        self._dof_states, self.dof_states = None, None
        self._q, self._qd = None, None
        self._ft_data, self.ft_data = None, None
        # self._forces, self.forces = None, None
        # self._jacobian, self.jacobian = None, None
        self.J_ee = None
        self._massmatrix, self.M = None, None
        self.default_dof_pos = None

        self.save_image_fpath = None
        self.frame_id = 0

    # ---- Scene construction -------------------------------------------------

    def _setup_scene(self):
        super()._setup_scene()

        # Task object(s)
        if self.cfg.use_valve:
            self.valve: RigidObject = RigidObject(self.cfg.valve_cfg)
            self.scene.rigid_objects["valve"] = self.valve

        if self.cfg.use_screwdriver:
            self.screwdriver: RigidObject = RigidObject(self.cfg.screwdriver_cfg)
            self.scene.rigid_objects["screwdriver"] = self.screwdriver

    def _get_rewards(self) -> torch.Tensor:
        return super()._get_rewards()

    def get_distance2goal(self):
        state = self.get_state()
        obj_state = state[:, -self.num_hand_dofs:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]
        obj_mat = obj_ori.as_matrix()
        distance2goal = tf.so3_relative_angle(torch.tensor(obj_mat), \
                                              torch.tensor(self.goal_pos.as_matrix()).repeat(self.num_envs, 1, 1),
                                              cos_angle=False).detach().cpu().abs()
        return distance2goal

    def get_state(self):
        arm_q = {
            'arm_q': self.hand_dof_pos
        }

        # Finger joint positions (map finger names to their joint indices in self.finger_to_joint_index)
        finger_q = {
            f"{finger}_q": self.hand.data.joint_pos[:, self.finger_to_joint_index[finger]]
            for finger in self.finger_names
        }

        # Finger end-effector positions via body links in Articulation data
        finger_ee_pos = {
            f"{finger}_pos": self.hand.data.body_link_pos_w[:, self.finger_ee_index[finger], :]
            for finger in self.finger_names
        }

        # Merge results
        results = {**finger_q, **finger_ee_pos, **arm_q}
        return results

    # ---- Reset logic --------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)

        # Randomize object pose above table/hand (special random pose for valve/screwdriver)
        def _rand_pose(N: int, z: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
            pos = torch.zeros((N, 3), device=self.device)
            pos[:, 0] = 0.0 + 0.02 * (torch.rand(N, device=self.device) - 0.5)
            pos[:, 1] = 0.0 + 0.02 * (torch.rand(N, device=self.device) - 0.5)
            pos[:, 2] = z
            # yaw-only random quaternion
            yaw = (torch.rand(N, device=self.device) - 0.5) * 2 * math.pi
            cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
            quat = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)
            return pos, quat

        if self.cfg.use_valve:
            pos, quat = _rand_pose(env_ids.numel())
            self.valve.write_root_pose_to_sim(pos, quat, env_ids=env_ids)
            self.valve.write_root_velocity_to_sim(
                torch.zeros((env_ids.numel(), 3), device=self.device),
                torch.zeros((env_ids.numel(), 3), device=self.device),
                env_ids=env_ids,
            )
        if self.cfg.use_screwdriver:
            pos, quat = _rand_pose(env_ids.numel())
            self.screwdriver.write_root_pose_to_sim(pos, quat, env_ids=env_ids)
            self.screwdriver.write_root_velocity_to_sim(
                torch.zeros((env_ids.numel(), 3), device=self.device),
                torch.zeros((env_ids.numel(), 3), device=self.device),
                env_ids=env_ids,
            )

# -----------------------------------------------------------------------------
# Cuboid Turning environment config
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
class AllegroCuboidTurningCfg(AllegroManipEnvCfg):
    # simulation / scene
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0/60.0, render_interval=2)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.5, replicate_physics=True)

    # cuboid object (placeholder uses an instanceable USD box asset; replace if you have USD export of the URDF)
    cuboid_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.31), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # action/observation sizes; action -> 16 allegro joint deltas
    action_space: int = 16
    observation_space: int = 64

    # default joint initial pose (derived from allegro.py default_dof_pos)
    default_q: float = 0.0

    def __post_init__(self):
        self.object_cfg = self.cuboid_cfg
        self.robot_cfg.init_state.pos = [-0.1, -0.025, 0.30]
        self.robot_cfg.init_state.rot = [0.258819, 0, 0, 0.9659258]
        #self.viewer.eye = [0.3, 0.3, 0.48]
        #self.viewer.look_at = [0.0, 0.0, 0.305]

# -----------------------------------------------------------------------------
# Helper: quaternion -> yaw (Z-up convention)
# -----------------------------------------------------------------------------

def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x,y,z,w) to yaw angle around Z.
    Expects shape (..., 4) in IsaacLab/Isaac Sim convention (x,y,z,w).
    Returns angles in radians shape (...,).
    """
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  assuming q=(x,y,z,w)
    x, y, z, w = q.unbind(-1)
    num = 2 * (w * z + x * y)
    den = 1 - 2 * (y * y + z * z)
    return torch.atan2(num, den)


# -----------------------------------------------------------------------------
# Environment implementation
# -----------------------------------------------------------------------------

class AllegroCuboidTurningEnv(AllegroManipEnv):
    def __init__(self, fingers: list[str], render_mode: Optional[str] = None, **kwargs):
        super().__init__(AllegroCuboidTurningCfg(fingers=fingers), render_mode=render_mode, **kwargs)

        # target yaw we want to achieve (per-env) — the task: rotate cuboid to this yaw
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
        super()._setup_scene()
        assert isinstance(self.cfg, AllegroCuboidTurningCfg)

        # spawn cuboid
        self.cuboid = RigidObject(self.cfg.cuboid_cfg)
        self.scene.rigid_objects["cuboid"] = self.cuboid

    def _get_rewards(self) -> torch.Tensor:
        return super()._get_rewards()

        assert len(self.actions.shape) == 2
        state = self.get_state()

        # goal cost
        obj_state = state[:, -self.obj_dof:]
        obj_pos = obj_state[:, :3]
        obj_ori = obj_state[:, 3:]

        cuboid_upright_cost = (obj_ori[:, 0] ** 2) + (obj_ori[:, 2] ** 2)
        reward -= 1000 * cuboid_upright_cost

        reward -= 10.0 * torch.sum((obj_pos - self.goal_pos[:, :3]) ** 2, dim=-1).to(self.device)
        distance2goal = self.get_distance2goal()
        reward -= 10 * torch.pow(distance2goal, 2).to(self.device)

        # dropp_flag = state[:, -4] < -0.1
        # reward -= 1000 * dropp_flag.to(self.device)
        # dropping cost
        reward -= 1e6 * ((state[:, -4] < -0.07) * state[:, -4]) ** 2

        # action_cost
        reward -= 50.0 * (torch.norm(self.actions, dim=-1) ** 2)

        # small penalty for high joint velocities
        reward -= 0.01 * torch.linalg.norm(self.hand.data.joint_vel, dim=-1, keepdims=True)
        return reward

    def get_state(self):
        results = super().get_state()
        results['cuboid_pos'] = self.cuboid.data.root_pos_w - self.scene.env_origins
        angles = euler_xyz_from_quat(self.object.data.root_quat_w)
        results['cuboid_quat'] = torch.tensor([angles[0], angles[1], angles[2]], device=self.device).reshape(1, len(angles))
        q = []
        for finger in self.finger_names:
            q.append(results[f'{finger}_q'])
        q.append(results['cuboid_pos'])
        q.append(results['cuboid_quat'])
        q = torch.cat(q, dim=1)
        results['q'] = q
        return q

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)

        N = env_ids.numel()
        # randomize cuboid pose, maintaining Z
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

        self.cuboid.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.cuboid.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids=env_ids)

        # set per-env target yaw (e.g. random target in [-pi,pi])
        self.target_yaw[env_ids] = (torch.rand(N, device=self.device) - 0.5) * 2 * math.pi

def get_env(task, img_save_dir, config, num_envs=1) -> AllegroManipEnv:
    if task == 'screwdriver_turning':
        """
        return AllegroScrewdriverTurningEnv(num_envs=num_envs,
                                           control_mode='joint_impedance',
                                            viewer=True,
                                            steps_per_action=60,
                                            friction_coefficient=1.0,
                                            device=config['sim_device'],
                                            video_save_path=img_save_dir,
                                            joint_stiffness=config['kp'],
                                            fingers=config['fingers'],
                                            gradual_control=config['gradual_control'],
                                            gravity=config['gravity'],
                                            )
        """
    elif task == 'valve_turning':
        """
        env = AllegroValveTurningEnv(num_envs=num_envs,
                                    control_mode='joint_impedance',
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                valve_type=config['object_type'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                random_robot_pose=config['random_robot_pose'],
                                )
        """
    elif task == 'cuboid_turning':
        return AllegroCuboidTurningEnv()
    elif task == 'cuboid_alignment':
        """
        return AllegroCuboidAlignmentEnv()
        """
    elif task == 'reorientation':
        """
        return AllegroReorientationEnv()
        """
    return None

# -----------------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--valve", action="store_true")
    parser.add_argument("--screwdriver", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()

    # Launch app
    app = AppLauncher(headless=args.headless).app

    # Build config
    cfg = AllegroManipEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.screwdriver:
        cfg.use_valve = False
        cfg.use_screwdriver = True

    # Create env
    env = AllegroManipEnv(cfg)

    # Rollout a few steps
    for _ in range(500):
        # random actions in [-1, 1]
        act = 2.0 * torch.rand((env.scene.num_envs, cfg.action_space), device=env.device) - 1.0
        obs, rew, done, info = env.step(act)

    app.close()
