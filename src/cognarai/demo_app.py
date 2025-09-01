#!/usr/bin/env python
# coding: utf-8

# Standard Library
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

# IsaacApp Launcher
# -> Must be always created first before importing Omniverse/Issac-related below
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Demo on spawning different objects in multiple environments.")
AppLauncher.add_app_launcher_args(parser) # [parser] is updated here with IsaacLab custom ones
args = parser.parse_args()
app_launcher = AppLauncher(args)

# Enable IsaacSim extensions
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot.manipulators")
enable_extension("isaacsim.robot_motion.motion_generation")

# Cognarai
from cognarai.isaac_app import IsaacApp
from cognarai.isaac import Isaac
isaac = Isaac()  # To init curobo configs

# !NOTE: All related to Isaac must be imported after [SimulationApp] creation above,
# which is done either by [IsaacWorld] or [AppLauncher]
from isaaclab.sim import (GroundPlaneCfg, DomeLightCfg, MultiAssetSpawnerCfg, MultiUsdFileCfg,
                          ConeCfg, CuboidCfg, SphereCfg,
                          PreviewSurfaceCfg, RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
    ArticulationCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class DemoInteractiveSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=[
                ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                SphereCfg(
                    radius=0.3,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=MassPropertiesCfg(mass=1.0),
            collision_props=CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    # articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=MultiUsdFileCfg(
            usd_path=[
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd"
            ],
            random_choice=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*shoulder_pan_joint": 0.0,
                ".*shoulder_lift_joint": -1.712,
                ".*elbow_joint": 1.712,
                ".*wrist_1_joint": 0.0,
                ".*wrist_2_joint": 0.0,
                ".*wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

def main(num_envs: int = 10):
    # IsaacApp
    app = IsaacApp(app_launcher,
                   interactive_scene_cfg=DemoInteractiveSceneCfg(num_envs=num_envs,
                                                                 env_spacing=2.0, replicate_physics=False),
                   args=args)

    # APP EXEC
    app.run()
    app.close()

if __name__ == "__main__":
    main()
