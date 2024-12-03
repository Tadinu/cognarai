#!/usr/bin/env python
# coding: utf-8

# Third Party
import torch

a = torch.zeros(
    4, device="cuda:0"
)  # this is necessary to allow isaac sim to use this torch instance

from pathlib import Path
from typing import List, Type, TYPE_CHECKING
import logging

logging.getLogger().setLevel(logging.INFO)

# Cognarai
from cognarai.isaac import Isaac
from cognarai.isaac_world import IsaacWorld
from cognarai.isaac_common import *

def main():
    # Init Isaac
    isaac = Isaac() # To init curobo configs

    # Init IsaacWorld
    # NOTE: THIS MUST BE A LIST OF DATA DIRECTORIES
    IsaacWorld.data_directory = [IsaacCommon().ISAAC_EXTERNAL_ASSETS_DIRECTORY]
    assert isinstance(IsaacWorld.data_directory, list)
    world = IsaacWorld()

    # Spawn robots & env objects
    #plane = Object("floor", ObjectType.ENVIRONMENT, "plane.urdf")
    #kitchen = Object("kitchen", ObjectType.ENVIRONMENT, "kitchen.urdf")
    ROBOT_MODEL_NAME = FRANKA_MODEL #UR10E_ALLEGRO_MODEL #PR2_MODEL #FRANKA_MODEL #UR5E_ROBOTIQ_2F_140_MODEL #UR10_WITH_SHORT_SUCTION_MODEL
    ROBOT_DESCRIPTION_DIR_NAME = "iiwa_allegro_description" if ROBOT_MODEL_NAME == IIWA_ALLEGRO_MODEL else "pr2_description"  if ROBOT_MODEL_NAME == PR2_MODEL else ""
    ROBOT_DESC_NAME = f"{ROBOT_DESCRIPTION_DIR_NAME}/{ROBOT_MODEL_NAME}.urdf"
    robot_name = f"{ROBOT_MODEL_NAME}_0"
    is_ur10_suction_robot = ROBOT_MODEL_NAME == UR10_WITH_SHORT_SUCTION_MODEL or ROBOT_MODEL_NAME == UR10_WITH_LONG_SUCTION_MODEL
    robot_position = [0.0, 0.0, 0.7] if is_ur10_suction_robot else [0.0, 0.0, 0.0]

    # SPAWN UR10 MOUNT
    if is_ur10_suction_robot:
        isaac.spawn_object(world.omni_world, UR10_MOUNT_MODEL, f"/World/ur10_mount", position=robot_position)

if __name__ == "__main__":
    main()
    #world.exit()
