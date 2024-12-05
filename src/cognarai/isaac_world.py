# used for delayed evaluation of typing until python 3.11 becomes mainstream
from __future__ import annotations

import logging
import threading
import time
from typing_extensions import Callable, Dict, List, Optional, Union
from pathlib import Path

# Numpy
import numpy as np
import matplotlib.pyplot as plt

# IsaacSim's third party path activation
try:
    import isaacsim
except ImportError:
    pass

# Omniverse/Isaac
from omni.isaac.kit import SimulationApp
# This is required before importing any other omni pkgs, including carb
# Ref: https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html#simulationapp
isaac_sim_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, PhysxSchema
import omni.usd
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.extensions import enable_extension
import omni.isaac.core.utils.numpy.rotations as rotation_utils
import omni.physx.scripts.utils as physxUtils

# Cognarai
from cognarai.isaac import Isaac, IsaacUSD
from cognarai.isaac_common import IsaacCommon, IsaacTaskId
from cognarai.omni_robot import OmniRobot

idle_tick = 0

class IsaacWorld(object):
    """
    This class represents a IsaacWorld, which represents a simulation world.
    """
    extension: str = ".usd, .usda, .usdc, .urdf"
    """
    Supported file extensions for spawnable object models
    """

    # Check is for sphinx autoAPI to be able to work in a CI workflow
    def __init__(self):
        self.isaac_sim_app = isaac_sim_app
        self.isaac_common = IsaacCommon()
        self.isaac = Isaac()

        # Some default settings
        self.set_gravity([0, 0, -9.8])
        self.run_experiment: Union[Callable, None] = None
        self.live_streamed = False

        # Init world
        self.init_world()

    def _init_extra_omniverse_extensions(self):
        ext_list = [
            "omni.kit.asset_converter",
            "omni.kit.tool.asset_importer",
            "omni.isaac.asset_browser",
        ]
        ext_list += ["omni.kit.livestream." + "True" if self.live_streamed else "False"]
        [enable_extension(x) for x in ext_list]
        isaac_sim_app.update()

    def init_world(self):
        self._init_omniverse_world()

    def _init_omniverse_world(self):
        # Omni world
        self.omni_world = omni.isaac.core.World(stage_units_in_meters=1.0)
        assert self.omni_world.scene
        Isaac().set_omni_world(self.omni_world)
        IsaacUSD().set_omni_world(self.omni_world)

        # Main world's stage
        self.stage = self.omni_world.stage
        # https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/stage/set-default-prim.html
        # It’s best practice to set the defaultPrim metadata on a Stage if the Stage’s root layer may be used as
        # a Reference or Payload.
        # Otherwise, consumers of your Stage are forced to provide a target prim when they create a Reference or Payload arc.
        # Even though the Usd.Stage.SetDefaultPrim() accepts any Usd.Prim, the default prim must be a top-level prim on the Stage.
        xform = self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(xform)
        #print(self.stage.GetRootLayer().ExportToString())

        # Ground
        self.omni_world.scene.add_default_ground_plane()

        # Camera
        self.main_camera = self.isaac.spawn_camera("/World/MainCamera",
                                                   position=np.array([0.0, 0.0, 25.0]),
                                                   orientation=rotation_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
                                                   add_motion_vector=False)

        # Extra Omni extensions
        #self._init_extra_omniverse_extensions()

    def exec_loop(self):
        task_id = IsaacTaskId.HANOI_TOWER
        if self.run_experiment:
            print('RUN EXPERIMENT')
            self.run_experiment()
        else:
            for robot_model_name, robots in self.isaac.robots.items():
                for robot in robots:
                    Isaac().init_robot_task(task_id, robot)

        while isaac_sim_app.is_running():
            self.omni_world.step(render=True)  # necessary to visualize changes
            for robot_model_name, robots in self.isaac.robots.items():
                for robot in robots:
                    Isaac().step_robot_task(task_id, robot)

            #self.step()

    def spawn_entity(self, entity_model_path: str, position: np.ndarray = [0,0,0], orientation: np.ndarray=[0,0,0,1]) -> int:
        if not entity_model_path:
            raise ValueError("Path to the object file is required.")

        # Load object configs, including kinematics
        entity_model_name = Path(entity_model_path).stem
        entity_prim = None
        entity_id = -1

        isaac = Isaac()
        if self.isaac_common.is_supported_robot_model(entity_model_name):
            isaac.load_entity_config(entity_model_path)
            entity_id = len(self.isaac.robots[entity_model_name]) if entity_model_name in self.isaac.robots else 0
            entity_prim_path = f"/World/{entity_model_name}{entity_id}"
            robot = isaac.spawn_robot(world=self.omni_world,
                                      robot_model_name=entity_model_name,
                                      robot_description_path=entity_model_path,
                                      robot_prim_path=entity_prim_path,
                                      position=position)
            entity_prim = robot.prim
        else:
            entity_id = len(self.isaac.objects[entity_model_name]) if entity_model_name in self.isaac.objects else 0
            entity_prim_path = f"/World/{entity_model_name}{entity_id}"
            object_entity = isaac.spawn_object(world=self.omni_world,
                                               object_model_path=entity_model_path,
                                               object_prim_path=entity_prim_path,
                                               position=position)
            entity_prim = object_entity.prim

        if entity_prim is None:
            return entity_id
        # Set entity pose
        UsdGeom.XformCommonAPI(entity_prim).SetTranslate(tuple(position))
        # prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(pose.position_as_list()), 0)
        UsdGeom.XformCommonAPI(entity_prim).SetRotate(self._quaternion_to_euler(Gf.Quatf(orientation[3],
                                                                                         orientation[0],
                                                                                         orientation[1],
                                                                                         orientation[2])))
        # prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(pose.orientation_as_list()), 0)
        return entity_id

    def _euler_to_quaternion(self, euler_angles):
        """
            Converts euler angles in quaternion

            Args:
                euler_angles: iterable list of euler angles in degrees

            Returns:
                the Gf.Quatf quaternion representation
        """

        euler_radians = [angle for angle in euler_angles]
        axes = [Gf.Vec3d(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
        quaternion = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        for angle, axis in zip(euler_radians, axes):
            rotation = Gf.Rotation(axis, angle)
            quaternion *= Gf.Quatf(rotation.GetQuat())
        return quaternion

    def _quaternion_to_euler(self, quaternion) -> tuple[float]:
        """
            Converts quaternion to euler angles

            Args:
                quaternion: a Gf.Quatf quaternion

            Returns:
                a list of Euler angles in degrees
        """

        rotation = Gf.Rotation(quaternion)
        axes = [Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()]
        euler_angles = rotation.Decompose(axes[0], axes[1], axes[2])
        return euler_angles[0], euler_angles[1], euler_angles[2]

    def remove_object_from_simulator(self, obj) -> None:
        self.stage.RemovePrim(obj.id)

    def add_constraint(self, name: str, parent_object_id: str, child_object_id: str) -> int:
        # https://github.com/ft-lab/omniverse_sample_scripts/blob/main/Physics/Joint
        # PhysicsArticulationRootAPI.py
        parent_obj = self.stage.GetPrimAtPath(parent_object_id, "Xform")
        child_obj = self.stage.GetPrimAtPath(child_object_id, "Xform")
        UsdPhysics.ArticulationRootAPI.Apply(parent_obj)
        UsdPhysics.RigidBodyAPI.Apply(child_obj)

        joint = physxUtils.createJoint(self.stage, name.lower().title(), parent_obj, child_obj)
        return joint

    def remove_constraint(self, constraint_id):
        self.stage.RemovePrim(constraint_id.GetPrimPath())

    def get_joint_position(self, joint_name: str) -> float:
        return 0
        isaac_idc = self.isaac.dynamic_control_interface
        #articulation = isaac_idc.get_articulation("/Articulation")
        #dof = isaac_idc.find_articulation_dof(articulation, joint_name)
        #return isaac_idc.get_dof_position(dof)

    def get_object_number_of_joints(self, obj) -> int:
        return len(self.get_object_joint_names(obj))

    def get_object_joint_names(self, obj) -> List[str]:
        return Isaac().get_entity_joint_names(Path(obj.path).stem)

    def get_link_pose(self, link) -> tuple(np.ndarray, np.ndarray):
        link_prim = self.stage.GetPrimAtPath(link.name)
        return link_prim.GetAttribute("xformOp:translate").Get(), link_prim.GetAttribute("xformOp:orient").Get()

    def get_object_link_names(self, obj) -> List[str]:
        return Isaac().get_entity_link_names(Path(obj.path).stem)

    def get_object_number_of_links(self, obj) -> int:
        return len(self.get_object_link_names(obj))

    def perform_collision_detection(self) -> None:
        #p.performCollisionDetection(physicsClientId=self.id)
        pass

    def get_object_contact_points(self, obj) -> List:
        """
        For a more detailed explanation of the
         returned list please look at:
         `PyBullet Doc <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#>`_
        """
        self.perform_collision_detection()
        return []

    def get_contact_points_between_two_objects(self, obj1, obj2) -> List:
        self.perform_collision_detection()
        return []

    def reset_joint_position(self, joint_name: str, joint_position: str) -> None:
        pass

    def reset_object_base_pose(self, obj, pose) -> None:
        pass

    def step(self):
        # Step omni world
        if isaac_sim_app.is_running():
            # NOTE: This invokes [isaac_sim_app.update()] here-in
            self.omni_world.step(render=True)
        else:
            return

        # Step-wise operations
        # Only proceed after user clicking `Play`
        global idle_tick
        if not self.omni_world.is_playing():
            if idle_tick % 100 == 0:
                idle_tick = 0
                print("**** Click Play to start simulation *****")
            idle_tick += 1
            return
        step_index = self.omni_world.current_time_step_index

        # Preliminary in-play setup for robots
        curobo = Isaac()
        if step_index <= 2:
            # self.omni_world.reset() -> THIS CAUSE Omni physics tensors api error: self._backend.is_homogeneous
            for robot_model_name, robots in self.isaac.robots.items():
                cspace = curobo.entity_configs[robot_model_name].kinematics.cspace
                j_names = cspace.joint_names
                default_config = cspace.retract_config
                for robot in robots:
                    idx_list = [robot.get_dof_index(x) for x in j_names]
                    robot.set_joint_positions(default_config.cpu().numpy(), idx_list)
                    robot._articulation_view.set_max_efforts(
                        values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                    )

    def get_object_pose(self, obj):
        prim = obj.id
        translate = prim.GetAttribute("xformOp:translate").Get()
        orient = prim.GetAttribute("xformOp:orient").Get()
        return translate, orient

    def set_link_color(self, link_name, rgba_color: Color):
        pass

    def get_link_color(self, link_name) -> Color:
        return self.get_colors_of_object_links(link_name)

    def get_colors_of_object_links(self, link_name) -> Color:
        return {}

    def get_object_axis_aligned_bounding_box(self, obj):
        pass

    def get_link_axis_aligned_bounding_box(self, link_name):
        pass

    def set_gravity(self, gravity_vector: List[float]) -> None:
        pass

    def disconnect_from_physics_server(self):
        isaac_sim_app.close()
        pass

    def join_threads(self):
        """
        Joins the GUI thread if it exists.
        """
        self.join_gui_thread_if_exists()

    def join_gui_thread_if_exists(self):
        if self._gui and issubclass(type(self._gui), threading.Thread):
            self._gui.join()

    def save_physics_simulator_state(self) -> int:
        return 0

    def restore_physics_simulator_state(self, state_id):
        pass

    def remove_physics_simulator_state(self, state_id: int):
        pass

    def get_images_for_target(self,
                              target_pose,
                              cam_pose,
                              size: Optional[int] = 256) -> List[np.ndarray]:
        self.main_camera.set_resolution((size, size))

        cam_position = cam_pose.pose.position
        cam_orientation = cam_pose.pose.orientation
        self.main_camera.set_world_pose(
            translation=[cam_position.x, cam_position.x, cam_position.z],
            orientation=[cam_orientation.w, cam_orientation.x, cam_orientation.y, cam_orientation.z]
        )
        current_frame = self.main_camera.get_current_frame()
        current_frame["normals"]
        current_frame["motion_vectors"]
        current_frame["occlusion"]
        current_frame["distance_to_image_plane"]
        current_frame["distance_to_camera"]
        current_frame["bounding_box_2d_tight"]
        current_frame["bounding_box_2d_loose"]
        current_frame["bounding_box_3d"]
        current_frame["semantic_segmentation"]
        current_frame["instance_id_segmentation"]
        current_frame["instance_segmentation"]
        current_frame["pointcloud"]

        rgba = self.main_camera.get_rgba()
        self.main_camera.get_rgb()
        self.main_camera.get_depth()
        self.main_camera.get_pointcloud()

        # Look at target_pose
        #points_2d = self.main_camera.get_image_coords_from_world_points(
        #    np.array([cube_3.get_world_pose()[0], cube_2.get_world_pose()[0]])
        #)

        if rgba:
            plt.imshow(self.main_camera.get_rgba()[:, :, :3])
            plt.show()
        print(self.main_camera.get_current_frame()["motion_vectors"])

    def add_text(self, text: str, position: List[float], orientation: Optional[List[float]] = None,
                 size: Optional[float] = None, color = None, life_time: Optional[float] = 0,
                 parent_object_id: Optional[int] = None, parent_link_id: Optional[int] = None) -> int:
        args = {}
        if orientation:
            args["textOrientation"] = orientation
        if size:
            args["textSize"] = size
        if life_time:
            args["lifeTime"] = life_time
        if parent_object_id:
            args["parentObjectUniqueId"] = parent_object_id
        if parent_link_id:
            args["parentLinkIndex"] = parent_link_id
        #return p.addUserDebugText(text, position, color.get_rgb(), physicsClientId=self.id, **args)
        return 0

    def remove_text(self, text_id: Optional[int] = None) -> None:
        pass

    def enable_joint_force_torque_sensor(self, obj, fts_joint_idx: int) -> None:
        pass

    def disable_joint_force_torque_sensor(self, obj, joint_id: int) -> None:
        pass

    def get_joint_reaction_force_torque(self, obj, joint_id: int) -> List[float]:
        return [0]

    def get_applied_joint_motor_torque(self, obj, joint_id: int) -> float:
        return 0
