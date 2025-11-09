from typing import Optional, Sequence
from pathlib import Path
import math

# Third-party
import torch

# Isaac Lab
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
# ISAACLAB_ASSET_FACTORY_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

# cognarai
from cognarai.mpc.mfr.allegro_cuboid_turning_env import AllegroCuboidTurningEnv, AllegroCuboidTurningCfg
from cognarai.mpc.mfr.mfr_spherical_6d_joint import mfr_add_free_joint


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Cuboid Alignment environment config
# -----------------------------------------------------------------------------

@configclass
class AllegroCuboidAlignmentCfg(AllegroCuboidTurningCfg):
    pass


# -----------------------------------------------------------------------------
# Environment implementation
# Original source: https://github.com/UM-ARM-Lab/MFR_benchmark
# -----------------------------------------------------------------------------

class AllegroCuboidAlignmentEnv(AllegroCuboidTurningEnv):
    def __init__(self, task_cfg: dict, render_mode: Optional[str] = None, **kwargs):
        super().__init__(task_cfg=task_cfg,
                         cfg=AllegroCuboidAlignmentCfg(fingers=task_cfg['fingers']), render_mode=render_mode, **kwargs)
        self.wall_pose = torch.tensor([0, -0.25, 0.19]).float().to(device=self.device)
        self.wall_dims = torch.tensor([0.1, 0.5, 0.12]).float().to(device=self.device)
