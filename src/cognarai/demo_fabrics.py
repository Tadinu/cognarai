#!/usr/bin/env python
# coding: utf-8

# Third Party
import torch

# Declare device for fabric
device_int = 0
device = f"cuda:{str(device_int)}"

# To allow isaac sim to use this torch instance
a = torch.zeros(4, device=device)
# Reduce print precision
torch.set_printoptions(precision=4)

# Enable the layers and stage windows in the UI
# Standard Library
import argparse

# Third Party
import numpy as np

import logging
logging.getLogger().setLevel(logging.INFO)

# Fabrics
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.visualization.robot_visualizer import RobotVisualizer
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

# Cognarai
# !NOTE: All related to Isaac must be imported after [IsaacWorld]
from cognarai.isaac_world import IsaacWorld
from cognarai.isaac import Isaac
from cognarai.isaac_common import *

# Parse arguments
parser = argparse.ArgumentParser(description='Kuka-Allegro fabric example.')
parser.add_argument('--batch_size', type=int, default=1, help='Specify batch size.')
parser.add_argument('--render', action='store_true', default=True, help='True to render fabric motion.')
parser.add_argument('--vis_col_spheres', action='store_true', default=False, help='True to visualize collision spheres of robot.')
parser.add_argument('--cuda_graph', action='store_true', default=True, help='True to enable graph capture of fabric.')
args = parser.parse_args()

# Settings
use_viz = args.render
render_spheres = args.vis_col_spheres
cuda_graph = args.cuda_graph
batch_size = args.batch_size

# This creates a world model that book keeps all the meshes
# in the world, their pose, name, etc.
print('Importing world')
world_filename = 'kuka_allegro_boxes'
max_objects_per_env = 20
world_model = WorldMeshesModel(batch_size=batch_size,
                               max_objects_per_env=max_objects_per_env,
                               device=device,
                               world_filename=world_filename)

# This reports back handles to the meshes which is consumed
# by the fabric for collision avoidance
object_ids, object_indicator = world_model.get_object_ids()

# Control rate and time settings
control_rate = 60.
timestep = 1./control_rate
total_time = 120.

def init_warp():
    # Set the warp cache directory based on device int
    #warp_cache_dir = ""
    initialize_warp(str(device_int))

def init_robot() -> tuple[BaseFabric, DisplacementIntegrator]:
    # Create Kuka-Allegro fabric palm pose and finger PCA action spaces
    kuka_allegro_fabric = KukaAllegroPoseFabric(batch_size, device, timestep, graph_capturable=cuda_graph)
    num_joints = kuka_allegro_fabric.num_joints

    # Create integrator for the fabric dynamics.
    kuka_allegro_integrator = DisplacementIntegrator(kuka_allegro_fabric)

    # Create starting states for the robot.
    # NOTE: first 7 angles are arm angles, last 16 angles are hand angles
    q = torch.tensor([-0.85, -0.50, 0.76, 1.25, -1.76, 0.90, 0.64,
                      0.0, 0.3, 0.3, 0.3,
                      0.0, 0.3, 0.3, 0.3,
                      0.0, 0.3, 0.3, 0.3,
                      0.72383858, 0.60147215, 0.33795027, 0.60845138], device=device)
    # Resize according to batch size
    q = q.unsqueeze(0).repeat(batch_size, 1).contiguous()
    # Start with zero initial velocities and accelerations
    qd = torch.zeros(batch_size, num_joints, device=device)
    qdd = torch.zeros(batch_size, num_joints, device=device)

    # The minimum and maximum values for the PCA targets, and initial targets
    hand_mins = torch.tensor([0.2475, -0.3286, -0.7238, -0.0192, -0.5532], device=device)
    hand_maxs = torch.tensor([3.8336, 3.0025, 0.8977, 1.0243, 0.0629], device=device)
    hand_targets = (hand_maxs - hand_mins) * torch.rand(batch_size, 5, device=device) + hand_mins

    # Palm target is (origin, Euler ZYX)
    palm_target = np.array([-0.6868, 0.0320, 0.6685, -2.3873, -0.0824, 3.1301])
    palm_target = torch.tensor(palm_target, device=device).expand((batch_size, 6)).float()

    # Get body sphere raddi
    body_sphere_radii = kuka_allegro_fabric.get_sphere_radii()

    # Get body sphere locations
    sphere_position, _ = kuka_allegro_fabric.get_taskmap("body_points")(q.detach(), None)

    return kuka_allegro_fabric, kuka_allegro_integrator

def init_fabrics(hand_targets, palm_target, q, qd, qdd, robot_fabric, fabric_integrator):
    # Graph capture
    g = None
    q_new = None
    qd_new = None
    qdd_new = None
    if cuda_graph:
        # NOTE: elements of inputs must be in the same order as expected in the set_features function
        # of the fabric
        inputs = [hand_targets, palm_target, "euler_zyx",
                  q.detach(), qd.detach(), object_ids, object_indicator]
        g, q_new, qd_new, qdd_new = \
            capture_fabric(robot_fabric, q, qd, qdd, timestep, fabric_integrator, inputs, device)

def main():
    # Init Isaac
    isaac = Isaac() # To init curobo configs

    # Init IsaacWorld
    assets_directory = IsaacCommon().ISAAC_EXTERNAL_ASSETS_DIRECTORY
    world = IsaacWorld()

    # Spawn robots & env objects
    #plane = Object("floor", ObjectType.ENVIRONMENT, "plane.urdf")
    #kitchen = Object("kitchen", ObjectType.ENVIRONMENT, "kitchen.urdf")
    ROBOT_MODEL_NAME = IIWA_ALLEGRO_MODEL
    ROBOT_DESCRIPTION_DIR_NAME = "iiwa_allegro_description" if ROBOT_MODEL_NAME == IIWA_ALLEGRO_MODEL else \
                                 "pr2_description" if ROBOT_MODEL_NAME == PR2_MODEL else assets_directory
    ROBOT_DESC_NAME = f"{ROBOT_DESCRIPTION_DIR_NAME}/{ROBOT_MODEL_NAME}.urdf"
    robot_position = [0.0, 0.0, 0.0]

    # SPAWN ROBOT
    world.spawn_entity(ROBOT_DESC_NAME, position=robot_position)

    # WORLD EXEC
    world.exec_loop()

    while world.isaac_sim_app.is_running():
        pass
if __name__ == "__main__":
    main()
    #world.exit()
