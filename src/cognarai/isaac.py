from __future__ import annotations

import pathlib

import logging

from pathlib import Path
from typing_extensions import Any, List, Dict, Type, Callable
import os
import logging

# Torch
import torch

a = torch.zeros(4, device="cuda:0")

# numpy
import numpy as np

# Omniverse kit
# NOTE: These imports are all required for the ensuing omni.isaac pkgs import
import carb
import carb.settings
from carb.settings import ISettings
import omni
import omni.usd
import omni.ext
import usdrt
from usdrt import Usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, PhysxSchema

# Omniverse-Isaac
import isaacsim
from isaacsim.core.api.world import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import VisualSphere
from isaacsim.core.api.tasks.base_task import BaseTask
from isaacsim.core.prims import XFormPrim, RigidPrim, Articulation
from isaacsim.core.api.articulations.articulation_gripper import ArticulationGripper
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.prims import create_prim, get_prim_at_path, is_prim_path_valid
import isaacsim.core.utils.numpy.rotations as rotation_utils
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.sensors.camera import Camera
#from isaacsim.robot_setup.assembler import RobotAssembler
#from isaacsim.asset.gen.omap.bindings import _omap as occupancy_map_manager
#from isaacsim.asset.gen.omap.utils import compute_coordinates, generate_image, update_location
from isaacsim.asset.importer.urdf import _urdf as omni_urdf
from isaacsim.asset.importer.mjcf import _mjcf as omni_mjcf

from omni.kit.viewport.utility import get_active_viewport
import omni.physx.scripts.physicsUtils as physicsUtils
import omni.physx.bindings._physx as PhysX
#from omni.physx import get_physx_simulation_interface

# Deprecated
from omni.isaac.dynamic_control import _dynamic_control as dynamic_control_manager
from omni.isaac.dynamic_control import utils as dynamic_control_utils

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig as CuroboRobotConfig
from curobo.types.file_path import ContentPath as CuroboContentPath
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver as CuroboIKSolver, IKSolverConfig as CuroboIKSolverConfig
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig

# Cognarai
from cognarai.isaac_common import *
from cognarai.omni_robot import OmniRobot
from cognarai.omni_robot_task import (OmniTargetFollowingTask, OmniTargetFollowingType, OmniPathPlanningTask,
                                      OmniSimpleStackingTask)
from cognarai.panda import Panda
from cognarai.curobo_robot_controller import CuroboBoxStackTask, CuroboRobotController
from cognarai.stacking_controller import StackingController

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class IsaacUSD(object, metaclass=Singleton):
    def __init__(self):
        """
        Create the singleton object of IsaacCommon class
        """
        self.isaac_common = IsaacCommon()
        self.omni_world = None
        self.omni_stage = None

    def set_omni_world(self, world: isaacsim.core.api.world.World):
        self.omni_world = world
        self.omni_stage = world.stage

    # https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/hierarchy-traversal/find-prims-by-type.html
    def get_prims_by_type(self, prim_type: Type[Usd.Typed]) -> List[Usd.Prim]:
        return [x for x in self.omni_stage.Traverse() if x.IsA(prim_type)] if self.omni_stage else []

    def get_all_prims(self) -> List[Usd.Prim]:
        return [x for x in self.omni_stage.Traverse()]

    def get_prim_by_name(self, prim_name: str) -> Optional[Usd.Prim]:
        if not is_prim_path_valid(prim_name):
            return None
        for prim in self.omni_stage.Traverse():
            if prim.GetName() == prim_name:
                return prim
        return None

    def get_child_prim(self, parent_prim: Usd.Prim, child_prim_name: str,
                       is_recursive: bool = True):
        result_prim = None
        for child_prim in parent_prim.GetChildren():
            child_prim_full_name = child_prim.GetName()
            print(f"{parent_prim.GetName()}:{child_prim_full_name}")
            if (child_prim.GetName() == child_prim_name or
                    child_prim_full_name.endswith(child_prim_name)):
                return child_prim
            # Depth-first search
            elif is_recursive:
                result_prim = self.get_child_prim(child_prim, child_prim_name)
                if result_prim:
                    return result_prim
        return result_prim

    def get_all_child_prims(self, parent_prim: Usd.Prim) -> List[Usd.Prim]:
        result_prims = parent_prim.GetChildren()
        for prim in result_prims:
            result_prims.extend(self.get_all_child_prims(prim))
        return result_prims


class Isaac(object, metaclass=Singleton):
    """
    Singleton class providing interface to Curobo accelerated compute-intensive services (ik-solver, collision check, motion control, etc.)

    Attributes
    ----------
    tensor_device : TensorDeviceType
        Device type for tensor computation
    config_paths: Dict[str, str]
        Dictionary of config file paths, keyed by entity model names
    entity_configs: Dict[str, RobotConfig]
        Dictionary of entity configs, keyed by entity model names
    ik_solvers: Dict[str, IKSolver]
        Dictionary of IK solvers, keyed by str(entity model name + base-link name + end-effector name)
    kinematics_models: Dict[str, CudaRobotModel]
        Dictionary of kinematics models, keyed by str(entity model name + end-effector name)
    """

    def __init__(self):
        """
        Create the singleton object of IsaacCommon class
        """
        self.EXAMPLE_ROBOT: Articulation = None
        self.EXAMPLE_GRIPPER: Articulation = None
        self.assembled_count: float = 0.0
        self.isaac_common = IsaacCommon()
        # Robots & objects spawned into the world
        self.robots: Dict[str, List[OmniRobot]] = {}
        self.objects: Dict[str, List[XFormPrim]] = {}

        self.omni_world = None
        self.omni_stage = None
        self.tensor_device = TensorDeviceType()
        self.config_paths: Dict[str, str] = \
            {model_name: os.path.join(self.isaac_common.CUROBO_EXTERNAL_CONFIGS_DIRECTORY, config_name)
             for model_name, config_name in self.isaac_common.CUROBO_MODEL_CONFIGS_FILENAMES.items()}

        self.entity_configs: Dict[str, CuroboRobotConfig] = {}
        self.ik_solvers: Dict[str, CuroboIKSolver] = {}
        self.kinematics_models: Dict[str, CudaRobotModel] = {}

        self.physx_interface: PhysX = None
        self.dynamic_control_interface = None
        self.occupancy_map_interface = None
        self.occupancy_map_generator = None
        self.robot_assembler = None
        self.init_tools()

        self.robot_tasks: Dict[IsaacTaskId, Tuple[Callable, Callable, Callable]]
        self.init_robot_tasks()

    def set_omni_world(self, world: isaacsim.core.api.world.World):
        self.omni_world = world
        self.omni_stage = world.stage

    def init_tools(self):
        # PhysX
        self.physx_interface = omni.physx.acquire_physx_interface()

        # Dynamic control
        self.dynamic_control_interface = dynamic_control_manager.acquire_dynamic_control_interface()
        # Dc freq, only settable after stage loading
        dynamic_control_utils.set_physics_frequency(60)

        # Occupancy map
        #self.init_occupancy_map_manager()

        # Robot Assembler
        #self.init_robot_assembler()

    def init_occupancy_map_manager(self):
        # Occupancy
        self.occupancy_map_interface = occupancy_map_manager.acquire_omap_interface()
        self.occupancy_map_generator = occupancy_map_manager.Generator(self.physx_interface,
                                                                       omni.usd.get_context().get_stage_id())
        generator = self.occupancy_map_generator
        generator.update_settings(0.05, 4, 5, 6)
        generator.set_transform((0, 0, 0), (-2.00, -2.00, 0), (2.00, 2.00, 0))
        generator.generate2d()
        assert len(generator.get_buffer()) == 0

    def init_robot_assembler(self):
        self.robot_assembler = RobotAssembler()

    def load_entity_config(self, urdf_file_path: str) -> None:
        """
        Load entity configuration and its kinematics models if supported by CuRobo
        :param urdf_file_path: path to the urdf file of the entity model
        :return: A tuple of entity config and its ik solver
        """
        urdf_exists = Path(urdf_file_path).exists()
        print('LOAD ENTITY CONFIG:', urdf_file_path, f"exists: {urdf_exists}")
        if not urdf_exists:
            logging.error(f"Entity model path does not exist: {urdf_file_path}")
            return
        entity_model_name = Path(urdf_file_path).stem
        if entity_model_name not in self.config_paths:
            logging.error(f"No Curobo yml config file registered yet for entity model: {entity_model_name}")
            return
        if entity_model_name in self.entity_configs:
            # Entity configs: already loaded
            return

        # [entity_yaml_path] -> [entity_yaml_config] -> [entity_config]
        entity_yaml_path = self.config_paths[entity_model_name]
        entity_yaml_config = load_yaml(entity_yaml_path)
        entity_kinematics = entity_yaml_config["robot_cfg"]["kinematics"]
        entity_kinematics["urdf_path"] = urdf_file_path # Overwrite one defined in yaml
        base_link = entity_kinematics["base_link"]
        if not base_link:
            logging.error(f"{entity_yaml_path}: No base_link configured")
            return
        ee_links = entity_kinematics["ee_links"]

        # Create an entity config based on base & ee links
        if self.isaac_common.is_supported_robot_model(entity_model_name):
            entity_config = CuroboRobotConfig.from_dict(entity_yaml_config, self.tensor_device)
            if not ee_links:
                logging.error(f"{entity_model_name}: No ee_links configured in {entity_yaml_path}")
                return
        else:
            entity_config = CuroboRobotConfig.from_basic(urdf_file_path, base_link, ee_links, self.tensor_device)

        # [entity_config]'s [kinematics]
        entity_content_path = CuroboContentPath(robot_config_root_path=self.isaac_common.CUROBO_EXTERNAL_CONFIGS_DIRECTORY,
                                                robot_asset_root_path=self.isaac_common.CUROBO_EXTERNAL_ASSETS_DIRECTORY,
                                                world_config_root_path=self.isaac_common.CUROBO_EXTERNAL_CONFIGS_DIRECTORY,
                                                world_asset_root_path=self.isaac_common.CUROBO_EXTERNAL_ASSETS_DIRECTORY,
                                                robot_urdf_absolute_path=urdf_file_path, robot_config_absolute_path=entity_yaml_path)
        entity_config.kinematics = CudaRobotModelConfig.from_content_path(entity_content_path, ee_links, self.tensor_device)
        self.entity_configs[entity_model_name] = entity_config

        # FK & IK_SOLVERS
        if self.isaac_common.is_supported_robot_model(entity_model_name):
            for ee_link in ee_links:
                # Create entity's kinematics model
                self.compute_forward_kinematics(entity_model_name, ee_link)

                # Load ik config
                ik_config_key = entity_model_name + base_link + ee_link
                if ik_config_key not in self.ik_solvers:
                    # Load ik config
                    ik_config = CuroboIKSolverConfig.load_from_robot_config(
                        entity_config,
                        None,
                        rotation_threshold=0.05,
                        position_threshold=0.005,
                        num_seeds=20,
                        self_collision_check=False,
                        self_collision_opt=False,
                        tensor_args=self.tensor_device,
                        use_cuda_graph=True,
                    )

                    # Create ik solver
                    self.ik_solvers[ik_config_key] = CuroboIKSolver(ik_config)

    def get_entity_link_names(self, entity_model_name) -> List[str]:
        """
        Get link names of an entity model
        :param entity_model_name: Entity model name, eg. "pr2", "kitchen"
        """
        return self.entity_configs[entity_model_name].kinematics.link_names \
            if entity_model_name in self.entity_configs else []

    def get_entity_ee_names(self, entity_model_name) -> List[str]:
        return self.entity_configs[entity_model_name].kinematics.generator_config.ee_links \
            if entity_model_name in self.entity_configs else []

    def get_entity_joint_names(self, entity_model_name) -> List[str]:
        """
        Get joint names of an entity model
        :param entity_model_name: Entity model name, eg. "pr2", "kitchen"
        """
        # all_articulated_joint_names
        joint_names = []
        ee_name_list = self.get_entity_ee_names(entity_model_name)
        for ee_name in ee_name_list:
            key = entity_model_name + ee_name
            if key in self.kinematics_models:
                joint_names.extend(self.kinematics_models[key].joint_names)
        return joint_names

    def get_robot_articulation_info(self, robot_prim_path) -> Optional[Tuple[Any, Any, Any]]:
        robot = self.dynamic_control_interface.get_articulation(robot_prim_path)
        if robot:
            return self.dynamic_control_interface.get_articulation_joint_count(robot), \
                self.dynamic_control_interface.get_articulation_dof_count(robot), \
                self.dynamic_control_interface.get_articulation_body_count(robot)
        else:
            return None

    def compute_forward_kinematics(self, entity_model_name: str, ee_link_name: str) -> Optional[CudaRobotModelState]:
        """
        Compute forward kinematics of an entity model
        :param entity_model_name: Entity model name, eg. "pr2", "kitchen"
        :param ee_link_name: End-effector link name
        """
        key = entity_model_name + ee_link_name
        if key in self.kinematics_models:
            kinematics_model = self.kinematics_models[key]
        else:
            if entity_model_name in self.entity_configs:
                kinematics_model = CudaRobotModel(self.entity_configs[entity_model_name].kinematics)
                self.kinematics_models[key] = kinematics_model
            else:
                logging.error(f"No configs found for entity model: {key}")
                return None
        q = torch.rand((10, kinematics_model.get_dof()), **(self.tensor_device.as_torch_dict()))
        return kinematics_model.get_state(q, ee_link_name)

    def spawn_object(self, world: isaacsim.core.api.world.World,
                     object_model_path: str,
                     object_prim_path: str,
                     position: np.array = np.array([0, 0, 0]),
                     orientation: np.array = np.array(euler_angles_to_quat([0, 0, 0], degrees=True))) -> Optional[RigidPrim]:
        # Only create object if [object_prim_path] does not exist yet
        # -> Also means [object_prim_path] must be unique!
        if is_prim_path_valid(object_prim_path):
            return None

        # Accept either full-path model name or relative one, of which the full path must have been registered
        object_model_name = Path(object_model_path).stem
        if os.path.isabs(object_model_path):
            if os.path.exists(object_model_path):
                object_usd_path = object_model_path
            else:
                logging.error(f"[{object_model_path}] does not exist!")
                return None
        else:
            if self.isaac_common.has_usd(object_model_name):
                object_usd_path = self.isaac_common.get_entity_default_full_usd_path(object_model_name)
            else:
                logging.error(f"Object [{object_model_name}] does not have USD model configured")
                return None

        # Create object prim
        object_usd_prim = create_prim(prim_path=object_prim_path, prim_type="Xform", position=position)
        add_reference_to_stage(usd_path=object_usd_path, prim_path=object_prim_path)

        # Gravity: disabled
        rigidBodyAPI = PhysxSchema.PhysxRigidBodyAPI.Apply(object_usd_prim)
        rigidBodyAPI.CreateDisableGravityAttr(True)

        # Collision: enabled
        collisionAPI = UsdPhysics.CollisionAPI.Apply(object_usd_prim)
        collisionAPI.CreateCollisionEnabledAttr().Set(True)

        # Mass: set mass and inertia
        massAPI = UsdPhysics.MassAPI.Apply(object_usd_prim)
        massAPI.CreateMassAttr().Set(0.01)
        #massAPI.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        #massAPI.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        # Make object rigid prim to set its world pose
        object = RigidPrim(name=f"{Path(object_prim_path).name}", prim_paths_expr=object_prim_path, masses=[0.1],
                           positions=np.array(position).reshape(1, 3),
                           orientations=np.array(orientation).reshape(1, 4))
        #object.set_world_poses(positions=([position]), orientations=([orientation]))

        # Add object to scene
        world.scene.add(object)
        logging.info(f"Object [{object.name}] has been spawned at path [{object_prim_path}]")

        # Append to [self.objects]
        if object_model_name not in self.objects:
            self.objects[object_model_name] = []
        self.objects[object_model_name].append(object)

        # Register [object] as rmp-obstacle to all robots
        for _, robots in self.robots.items():
            for robot in robots:
                robot.register_obstacle_prim(object)

        return object

    def spawn_robot(self, world: isaacsim.core.api.world.World,
                    robot_model_name: str,
                    robot_description_path: str,
                    robot_prim_path: str,
                    subroot: str = "",
                    position: np.array = np.array([0, 0, 0]),
                    orientation: np.array = np.array([1, 0, 0, 0]),
                    visualize_robot_spheres: bool = False) -> Optional[Robot]:
        if not Path(robot_description_path).exists():
            assert False, f"{robot_description_path} does not exist!"
        is_urdf_robot = robot_description_path.endswith(".urdf")
        is_mjcf_robot = robot_description_path.endswith(".mjcf") or robot_description_path.endswith(".xml")

        robot_description_interface = omni_urdf.acquire_urdf_interface() if is_urdf_robot \
            else omni_mjcf.acquire_mjcf_interface() if is_mjcf_robot else None
        import_config = omni_urdf.ImportConfig() if is_urdf_robot \
            else omni_mjcf.ImportConfig() if is_mjcf_robot else None

        if robot_description_interface is None:
            logging.error(f"Unrecognized robot description path extension: {Path(robot_description_path).suffix}")
            return None

        # Import configs
        import_config.merge_fixed_joints = False  # NOTE This must be false for ur, franka
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.make_default_prim = False
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        import_config.default_drive_strength = 20000
        import_config.default_position_drive_damping = 500
        import_config.default_drive_type = omni_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.distance_scale = 1
        import_config.density = 0.0

        robot_name = Path(robot_prim_path).name
        robot_description_dir_path, robot_description_file_name = os.path.split(robot_description_path)
        robot_base_body_model_name = self.isaac_common.get_robot_base_body_model_name(robot_model_name)
        if not self.isaac_common.has_usd(robot_model_name) and not self.isaac_common.has_usd(robot_base_body_model_name):
            parse_robot_description = robot_description_interface.parse_urdf if is_urdf_robot else \
                robot_description_interface.parse_mjcf
            robot_description = parse_robot_description(robot_description_dir_path,
                                                        robot_description_file_name,
                                                        import_config)
            robot_prim_path = robot_description_interface.import_robot(
                robot_description_dir_path,
                robot_description_file_name,
                robot_description,
                import_config,
                subroot,
            )

        # NOTE: Articulation from 2023.1.1 must be put at base link, otherwise "'NoneType' object has no attribute 'is_homogeneous'" arise upon world.reset()
        # https://forums.developer.nvidia.com/t/enhancements-to-urdf-importer-articulation-in-version-2023-1-1-ensure-consistent-root-link-designation-for-improved-simulation-accuracy/283616
        is_articulation_supported = self.isaac_common.is_supported_articulation_model(robot_model_name)
        robot_kinematics_config = self.entity_configs[robot_model_name].kinematics.kinematics_config
        robot_articulation_root_path = f"{robot_prim_path}/{robot_kinematics_config.base_link}" \
            if is_articulation_supported else robot_prim_path

        if (self.isaac_common.has_builtin_gripper_model(robot_model_name) and
                self.isaac_common.is_modular_robot_models(robot_model_name)):
            robot = self.spawn_assembled_robot(robot_model_name=robot_model_name,
                                               description_path="",
                                               robot_prim_path=robot_prim_path,
                                               robot_name=robot_name,
                                               articulation_root_path=robot_articulation_root_path,
                                               end_effector_frame=self.isaac_common.get_robot_ee_mount_frame_name(robot_model_name),
                                               gripper_mount_frame=self.isaac_common.get_gripper_mount_frame_name(robot_model_name))

            self.spawn_example_assembled_robot()
        else:
            robot_class = self.isaac_common.get_robot_class(robot_model_name)
            robot = robot_class(
                robot_model_name=robot_model_name,
                description_path=robot_description_path,
                prim_path=robot_prim_path,
                articulation_root_path=robot_articulation_root_path,
                name=robot_name,
                position=position,
                orientation=orientation,
                end_effector_prim_name=f"{robot_prim_path}/{robot_kinematics_config.ee_links[0]}"
                if robot_model_name != PR2_MODEL else robot_kinematics_config.ee_links[0]
            )
            world.scene.add(robot)
        logging.info(f"Robot [{robot.robot_unique_name}] has been spawned at path [{robot.prim.GetPrimPath()}]")

        # https://forums.developer.nvidia.com/t/cant-create-simulation-view/252875/2
        # NOTE: Don't call robot.initialize(), which raises `Failed to create simulation view backend`, which seems to only
        # be creatable async
        stage = robot.prim.GetStage()
        articulation_root = stage.GetPrimAtPath(robot_articulation_root_path)
        if is_articulation_supported and articulation_root:
            #mass = UsdPhysics.MassAPI(articulation_root)
            #massAPI.GetMassAttr().Set(0.1)
            massAPI = UsdPhysics.MassAPI.Apply(articulation_root)
            massAPI.CreateMassAttr(0.1)

        # Visualize collision spheres (configured in <robot_model_name>.yml)
        if visualize_robot_spheres:
            robot_kin_model = CudaRobotModel(self.entity_configs[robot_model_name].kinematics)
            default_config = robot_kin_model.cspace.retract_config

            ee = robot_kinematics_config.ee_links[0]
            sph_list = robot_kin_model.get_robot_as_spheres(default_config, ee)
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    name=f"{robot_model_name}_sphere_{si}",
                    prim_path=f"/World/curobo/{robot_model_name}_sphere_{si}",
                    position=np.ravel(s.position) + position,
                    radius=float(s.radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                world.scene.add(sp)

        # Append to [self.robots]
        if robot_model_name not in self.robots:
            self.robots[robot_model_name] = []
        self.robots[robot_model_name].append(robot)
        return robot

    def spawn_example_assembled_robot(self, robot_model_name: str = UR10E_MODEL,
                                      gripper_model_name: str = ROBOTIQ_2F85_MODEL):
        base_robot_path = f"/World/{robot_model_name}"
        attach_robot_path = f"/World/{gripper_model_name}"
        add_reference_to_stage(usd_path=self.isaac_common.get_entity_default_full_usd_path(robot_model_name),
                               prim_path=base_robot_path)
        XFormPrim(base_robot_path).set_world_pose(np.array([1.0, 0.0, 3.0]))
        add_reference_to_stage(usd_path=self.isaac_common.get_entity_default_full_usd_path(gripper_model_name),
                               prim_path=attach_robot_path)
        XFormPrim(attach_robot_path).set_world_pose(np.array([-1.0, 0.0, 3.0]))
        base_robot_mount_frame = f"/{self.isaac_common.get_robot_ee_mount_frame_name(robot_model_name)}"
        attach_robot_mount_frame = f"/{self.isaac_common.get_gripper_mount_frame_name(gripper_model_name)}"

        fixed_joint_offset = np.array([0.0, 0.0, 0.2])
        fixed_joint_orient = np.array([0.956, 0.0, -0.0, 0.2935])

        robot_assembler = RobotAssembler()
        assembled_body = robot_assembler.assemble_articulations(
            base_robot_path,
            attach_robot_path,
            base_robot_mount_frame,
            attach_robot_mount_frame,
            fixed_joint_offset,
            fixed_joint_orient,
            mask_all_collisions=True,
            single_robot=False,
        )

        assembled_body.set_fixed_joint_transform(np.array([0.0, 0.0, -0.02]),
                                                 np.array([1.0, 0.0, 0.0, 0.0]))

        # The robots are controlled independently from each other
        self.EXAMPLE_ROBOT = Articulation(prim_path=base_robot_path, name=f"{robot_model_name}_example")
        self.EXAMPLE_GRIPPER = Articulation(prim_path=attach_robot_path,
                                            name=f"{self.isaac_common.get_robot_gripper_model_name(robot_model_name)}_example")
        self.omni_world.scene.add(self.EXAMPLE_ROBOT)
        self.omni_world.scene.add(self.EXAMPLE_GRIPPER)

    def spawn_assembled_robot(self,
                              robot_model_name: str,
                              description_path: str,
                              robot_prim_path: str,
                              robot_name: str,
                              articulation_root_path: str,
                              end_effector_frame: str,
                              gripper_mount_frame: str,
                              as_single_articulation: bool = False) \
            -> Optional[OmniRobot]:
        assert self.isaac_common.has_builtin_gripper_model(robot_model_name)

        # 1- Spawn robot body to scene
        if is_prim_path_valid(robot_prim_path):
            logging.error(f"[spawn_assembled_robot()][{robot_model_name}:{robot_prim_path}] already exists")
            return None
        robot_base_body_model_name = self.isaac_common.get_robot_base_body_model_name(robot_model_name)
        assert add_reference_to_stage(usd_path=self.isaac_common.get_entity_default_full_usd_path(robot_base_body_model_name),
                                      prim_path=robot_prim_path)
        XFormPrim(robot_prim_path).set_world_pose(np.array([1.0, 0.0, 0.5]))

        # 2- Spawn gripper to scene
        gripper_model_name = self.isaac_common.get_robot_gripper_model_name(robot_model_name)
        gripper_prim_path = find_unique_string_name(
            initial_name=f"/World/{gripper_model_name}", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        gripper_usd_path = self.isaac_common.get_entity_default_full_usd_path(gripper_model_name)
        if not gripper_usd_path:
            logging.error(f"[spawn_assembled_robot()][{gripper_model_name}] has no gripper usd")
            return None
        assert add_reference_to_stage(usd_path=gripper_usd_path, prim_path=gripper_prim_path)
        XFormPrim(gripper_prim_path).set_world_pose(np.array([-1.0, 0.0, 0.5]))

        # Assemble robot-body + gripper
        base_robot_mount_frame = f"/{end_effector_frame}"
        attach_robot_mount_frame = f"/{gripper_mount_frame}"
        fixed_joint_offset = np.array([0.0, 0.0, 0.2])
        fixed_joint_orient = np.array([0.956, 0.0, -0.0, 0.2935])

        assembled_body = self.robot_assembler.assemble_articulations(
            robot_prim_path,
            gripper_prim_path,
            base_robot_mount_frame,
            attach_robot_mount_frame,
            fixed_joint_offset,
            fixed_joint_orient,
            mask_all_collisions=True,
            single_robot=as_single_articulation
        )

        if "allegro" in robot_model_name:
            assembled_body.set_fixed_joint_transform(np.array([0.0, 0.0, -0.15]),
                                                     np.array([0.956, 0.0, -0.0, 0.2935]))
        else:
            assembled_body.set_fixed_joint_transform(np.array([0.0, 0.0, -0.02]),
                                                     np.array([1.0, 0.0, 0.0, 0.0]))

        robot = OmniRobot(robot_model_name=robot_model_name,
                          description_path=description_path,
                          end_effector_prim_name=f"{robot_prim_path}/{end_effector_frame}",
                          prim_path=robot_prim_path,
                          name=robot_name,
                          articulation_root_path=articulation_root_path,
                          attach_extra_gripper=False,
                          assembled_body=assembled_body,
                          rmp_policy_name="RMPflow",
                          rmp_policy_with_gripper_name="RMPflow")

        robot.gripper_articulation = OmniRobot(robot_model_name=gripper_model_name,
                                               description_path=description_path,
                                               end_effector_prim_name=f"{gripper_prim_path}/right_inner_finger",
                                               prim_path=gripper_prim_path,
                                               name=f"{robot_name}_gripper_articulation",
                                               articulation_root_path=gripper_prim_path,
                                               gripper_articulation_root_path=gripper_prim_path,
                                               attach_extra_gripper=True,
                                               assembled_body=assembled_body,
                                               rmp_policy_name="RMPflow",
                                               rmp_policy_with_gripper_name="RMPflow")

        # Add assembled_robot to scene
        # NOTE: By being added to scene, it will be auto-initialized later upon omni_world.reset()
        self.omni_world.scene.add(robot)
        self.omni_world.scene.add(robot.gripper_articulation)
        return robot

    def init_robot_tasks(self):
        self.robot_tasks = {
            IsaacTaskId.TARGET_FOLLOWING: (lambda robot: f"{robot.name}_target_follow_task",
                                           self.init_target_following, self.step_target_following),
            IsaacTaskId.PATH_PLANNING: (lambda robot: f"{robot.name}_path_plan_task",
                                        self.init_path_planning, self.step_path_planning),
            IsaacTaskId.PICK_PLACE_PANDA: (lambda robot: f"{robot.name}_pick_place_task",
                                           self.init_panda_picking_task, self.step_panda_picking),
            IsaacTaskId.PICK_PLACE_CUROBO: (lambda robot: f"{robot.name}_pick_place_curobo_task",
                                            self.init_curobo_pick_place, self.step_curobo_pick_place),
            IsaacTaskId.SIMPLE_STACKING: (lambda robot: f"{robot.name}_simple_stacking_task",
                                          self.init_simple_stacking, self.step_simple_stacking),
            IsaacTaskId.HANOI_TOWER: (lambda robot: f"{robot.name}_hanoi_tower_task",
                                      self.init_hanoi_tower, self.step_hanoi_tower),
            IsaacTaskId.MPC_CUROBO: (lambda robot: f"{robot.name}_mpc_curobo_task",
                                     self.init_curobo_mpc, self.step_curobo_mpc)
        }

    def add_robot_task(self, task: BaseTask) -> None:
        if self.omni_world:
            # 1- Add task to world
            self.omni_world.add_task(task)
            # 2- Reset world, and physics_sim_view, articulation_view, etc.
            # 2.1- Also invokes task's [set_up_scene()]
            # 2.2- Initialize robots (due to scene finalizing)
            self.omni_world.reset()

            # 3- Register obstacles in task (can only be done after [omni_world.reset()]
            if isinstance(task, OmniPathPlanningTask):
                # Also auto-register added obstacles to task's robot here-in!
                # NOTE: Cannot invoke this in [set_up_scene()] (for unclear reason)
                # Also even it can, it's not ok due to task's robot also registering these obstacles here-in,
                # which requires the robot's physics sim view to have been initialized in advance in [omni_world.reset()]
                task.add_obstacles()
        else:
            assert False, "Omni Isaac world must be present for robot task!"

    def get_robot_task_name(self, task_id: IsaacTaskId, robot: OmniRobot) -> str:
        return self.robot_tasks[task_id][0](robot)

    def get_robot_task(self, task_id: IsaacTaskId, robot: OmniRobot) -> BaseTask:
        return self.omni_world.get_task(self.get_robot_task_name(task_id, robot))

    def init_robot_task(self, task_id: IsaacTaskId, robot: OmniRobot, target_name: Optional[str] = None):
        if task_id in self.robot_tasks:
            init_task = self.robot_tasks[task_id][1]
            if init_task:
                if task_id == IsaacTaskId.TARGET_FOLLOWING or task_id == IsaacTaskId.PATH_PLANNING:
                    init_task(robot, target_name)
                else:
                    init_task(robot)

    def step_robot_task(self, task_id: IsaacTaskId, robot: OmniRobot, target_name: Optional[str] = None):
        robot.update()
        if task_id in self.robot_tasks:
            step_task = self.robot_tasks[task_id][2]
            if step_task:
                step_task(robot)

    def init_target_following(self, robot: OmniRobot, target_name: Optional[str] = None):
        target_follow_task = robot.create_target_following_task(
            task_name=self.get_robot_task_name(IsaacTaskId.TARGET_FOLLOWING, robot),
            following_type=OmniTargetFollowingType.RMP,
            target_name=target_name
        )
        # This invokes [self.omni_world.reset()] -> task's [set_up_scene()]
        self.add_robot_task(target_follow_task)

    def step_target_following(self, robot: OmniRobot):
        if self.omni_world.is_playing():
            target_follow_task: OmniTargetFollowingTask = self.get_robot_task(IsaacTaskId.TARGET_FOLLOWING, robot)
            is_using_rmp = target_follow_task.is_using_rmp()
            if self.omni_world.current_time_step_index == 0:
                self.omni_world.reset()
                if is_using_rmp:
                    robot.rmp_flow_controller.reset()

            if is_using_rmp:
                observations = self.omni_world.get_observations()[target_follow_task.target_name]
                actions = robot.rmp_flow_controller.forward(
                    target_end_effector_position=observations["position"],
                    target_end_effector_orientation=observations["orientation"],
                )
                robot.apply_action(actions)
            else:
                robot.follow_target(target_follow_task.target_name)

    def init_path_planning(self, robot: OmniRobot, target_name: Optional[str] = None):
        #NOTE: Here is only for the task env to be initialized, robot's controllers may not have been initialized yet!
        path_planning_task = robot.create_path_planning_task(
            task_name=self.get_robot_task_name(IsaacTaskId.PATH_PLANNING, robot),
            target_name=target_name
        )
        # This invokes [self.omni_world.reset()] -> task's [set_up_scene()], adding obstacles
        self.add_robot_task(path_planning_task)

    def step_path_planning(self, robot: OmniRobot):
        if self.omni_world.is_playing():
            if not robot.path_rrt_controller or not robot.path_rrt_controller.get_path_planner():
                return
            if self.omni_world.current_time_step_index == 0:
                self.omni_world.reset()
                robot.path_rrt_controller.reset()
            path_plan_task: OmniPathPlanningTask = self.get_robot_task(IsaacTaskId.PATH_PLANNING, robot)
            observations = self.omni_world.get_observations()[path_plan_task.target_name]
            actions = robot.path_rrt_controller.forward(
                target_end_effector_position=observations["position"],
                target_end_effector_orientation=observations["orientation"],
            )
            articulation_controller = robot.get_articulation_controller()
            articulation_controller.set_gains(*robot.get_custom_gains())
            articulation_controller.apply_action(actions)

    def init_simple_stacking(self, robot: OmniRobot):
        stacking_task = robot.create_simple_stacking_task(
            task_name=self.get_robot_task_name(IsaacTaskId.SIMPLE_STACKING, robot),
            target_position=np.array([0.5, 0.5, 0]) / get_stage_units()
        )
        # This invokes [self.omni_world.reset()] -> task's [set_up_scene()]
        self.add_robot_task(stacking_task)

        # NOTE: This stacking controller cannot be created right in [robot.create_simple_stacking_task()] for reasons:
        # (1) Cubes will be created in set_up_scene(), which is invoked by world.reset() upon [self.add_robot_task()]
        # (2) StackingController invoke PickingController, which requires [robot.rmp_flow_controller], which is only
        # initialized upon [robot.initialize()] in [world.reset()]
        robot.task_controller = StackingController(
            name=f"{robot.robot_unique_name}_stacking_controller",
            gripper=robot.gripper_articulation.gripper if robot.gripper_articulation else robot.gripper,
            robot_articulation=robot,
            picking_order_cube_names=stacking_task.get_cube_names(),
            robot_observation_name=robot.robot_unique_name
        )

    def step_simple_stacking(self, robot: OmniRobot):
        if self.omni_world.is_playing():
            if self.omni_world.current_time_step_index == 0:
                #NOTE: This invokes current tasks' set_up_scene()
                self.omni_world.reset()
                robot.task_controller.reset()
            observations = self.omni_world.get_observations()

            current_event = robot.pick_place_controller.get_current_event()
            actions = robot.task_controller.forward(observations=observations,
                                                    end_effector_offset=np.array([0.0, 0.0, 0.02]))

            if robot.gripper_articulation and (current_event == 3 or current_event == 7):
                robot.gripper_articulation.apply_action(actions)
            else:
                robot.apply_action(actions)

            #self.assembled_count+=0.001
            #if self.assembled_count >=3:
            #    self.assembled_count = 0.0
            #if self.EXAMPLE_ROBOT:
            #    base_robot_joint_target = np.array([0.3, self.assembled_count, 0, self.assembled_count, 0, 0])
            #    self.EXAMPLE_ROBOT.apply_action(ArticulationAction(base_robot_joint_target))
            #if self.EXAMPLE_GRIPPER:
            #    gripper_dofs_num = self.EXAMPLE_GRIPPER._articulation_view.num_dof
            #    gripper_joint_target = np.zeros(gripper_dofs_num)
            #    for i in range(gripper_dofs_num):
            #        gripper_joint_target[i] += self.assembled_count
            #    self.EXAMPLE_GRIPPER.apply_action(ArticulationAction(gripper_joint_target))

    def init_hanoi_tower(self, robot: OmniRobot):
        hanoi_tower_task = robot.create_hanoi_tower_task(
            task_name=self.get_robot_task_name(IsaacTaskId.HANOI_TOWER, robot)
        )
        # This invokes [self.omni_world.reset()] -> task's [set_up_scene()]
        self.add_robot_task(hanoi_tower_task)

        # NOTE: This stacking controller cannot be created right in [robot.create_simple_stacking_task()] for reasons:
        # (1) Cubes will be created in set_up_scene(), which is invoked by world.reset() upon [self.add_robot_task()]
        # (2) StackingController invoke PickingController, which requires [robot.rmp_flow_controller], which is only
        # initialized upon [robot.initialize()] in [world.reset()]
        robot.task_controller = StackingController(
            name=f"{robot.robot_unique_name}_stacking_controller",
            gripper=robot.gripper_articulation.gripper if robot.gripper_articulation else robot.gripper,
            robot_articulation=robot,
            picking_order_cube_names=hanoi_tower_task.get_disk_names(),
            robot_observation_name=robot.robot_unique_name
        )

    def step_hanoi_tower(self, robot: OmniRobot):
        if self.omni_world.is_playing():
            if self.omni_world.current_time_step_index == 0:
                #NOTE: This invokes current tasks' set_up_scene()
                self.omni_world.reset()
                robot.task_controller.reset()
            observations = self.omni_world.get_observations()

            current_event = robot.pick_place_controller.get_current_event()
            actions = robot.task_controller.forward(observations=observations,
                                                    end_effector_offset=np.array([0.0, 0.0, 0.02]),
                                                    end_effector_orientation=euler_angles_to_quat(
                                                        np.array([0.0, np.pi/2.0, 0.0])))

            if robot.gripper_articulation and (current_event == 3 or current_event == 7):
                robot.gripper_articulation.apply_action(actions)
            else:
                robot.apply_action(actions)

    def init_curobo_mpc(self, robot: OmniRobot):
        robot.curobo_robot_controller = CuroboRobotController(robot=robot, world=self.omni_world,
                                                              task=None,
                                                              constrain_grasp_approach=False)

    def step_curobo_mpc(self, robot: OmniRobot):
        robot.curobo_robot_controller.step_mpc()

    def init_curobo_pick_place(self, robot: OmniRobot):
        if self.omni_world is None:
            logging.error("Isaac world has not been created yet")
            return
        #robot_prim_path = "/World/Franka/panda_link0"
        self.curobo_ignored_substring = ["Franka", "TargetCube", "material", "Plane"]

        # Create pick-place task
        robot.curobo_pick_place_task = CuroboBoxStackTask(robot=robot,
                                                          name=self.get_robot_task_name(IsaacTaskId.PICK_PLACE_CUROBO, robot))
        # Reset articulation_view here-in
        # This invokes [self.omni_world.reset()] -> task's [set_up_scene()]
        self.add_robot_task(robot.curobo_pick_place_task)
        robot_articulation_controller = robot.get_articulation_controller()
        print(robot, robot._articulation_view, robot.get_joints_state())
        robot_articulation_controller.initialize(None, robot._articulation_view)

        # Create robot controller
        robot.curobo_robot_controller = CuroboRobotController(robot=robot, world=self.omni_world,
                                                              task=robot.curobo_pick_place_task,
                                                              constrain_grasp_approach=False)

        # Create camera view
        set_camera_view(eye=[2, 0, 1], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")
        self.curobo_wait_steps = 8

        # Config solver vel/pos iteration & physics solver
        robot.set_solver_velocity_iteration_count(4)
        robot.set_solver_position_iteration_count(124)
        print(
            self.omni_world._physics_context.get_solver_type(),
            robot.get_solver_position_iteration_count(),
            robot.get_solver_velocity_iteration_count(),
        )
        self.omni_world._physics_context.set_solver_type("TGS")
        self.initial_steps = 100
        ################################################################
        print("Start simulation...")
        print("Use GPU Pipeline", self.omni_world._physics_context.use_gpu_pipeline)
        robot.enable_gravity()
        print(robot_articulation_controller.get_gains())
        print(robot_articulation_controller.get_max_efforts())
        robot.enable_gravity()
        print("**********************")
        robot_articulation_controller.set_gains(
            kps=np.array(
                [100000000, 6000000.0, 10000000, 600000.0, 25000.0, 15000.0, 50000.0, 6000.0, 6000.0]
            )
        )

        robot_articulation_controller.set_max_efforts(
            values=np.array([100000, 52.199997, 100000, 52.199997, 7.2, 7.2, 7.2, 50.0, 50])
        )

        print("Updated gains:")
        print(robot_articulation_controller.get_gains())
        print(robot_articulation_controller.get_max_efforts())
        # exit()
        if robot.gripper:
            robot.gripper.open()
        for _ in range(self.curobo_wait_steps):
            self.omni_world.step(render=True)
        robot.curobo_pick_place_task.reset()
        self.curobo_task_finished = False
        observations = self.omni_world.get_observations()
        robot.curobo_pick_place_task.get_pick_position(observations)
        self.curobo_pick_place_loop_count = 0

    def step_curobo_pick_place(self, robot: OmniRobot):
        self.curobo_pick_place_loop_count += 1

        if self.curobo_task_finished or self.curobo_pick_place_loop_count < self.initial_steps:
            return

        if not robot.curobo_robot_controller.init_curobo:
            robot.curobo_robot_controller.reset(self.curobo_ignored_substring, robot.articulation_root_path)

        step_index = self.omni_world.current_time_step_index
        observations = self.omni_world.get_observations()
        sim_js = robot.get_joints_state()
        ee_links = self.entity_configs[robot.robot_model_name].kinematics.kinematics_config.ee_links \
            if robot.robot_model_name in self.entity_configs else []
        if not ee_links:
            logging.error(f"{robot.robot_model_name} have empty ee_links configured in yml")
            return
        ee = ee_links[0]

        if robot.curobo_robot_controller.reached_target(observations):
            if robot.gripper.get_joint_positions()[0] < 0.035:  # reached placing target
                robot.gripper.open()
                for _ in range(self.curobo_wait_steps):
                    self.omni_world.step(render=True)
                robot.curobo_robot_controller.detach_obj()
                robot.curobo_robot_controller.update(
                    self.curobo_ignored_substring, robot.articulation_root_path
                )  # update world collision configuration
                self.curobo_task_finished = robot.curobo_pick_place_task.update_task()
                if self.curobo_task_finished:
                    print("\nTASK DONE\n")
                    for _ in range(self.curobo_wait_steps):
                        self.omni_world.step(render=True)
                    return
                else:
                    robot.curobo_pick_place_task.get_pick_position(observations)

            else:  # reached picking target
                if robot.gripper:
                    robot.gripper.close()
                for _ in range(self.curobo_wait_steps):
                    self.omni_world.step(render=True)
                sim_js = robot.get_joints_state()
                robot.curobo_robot_controller.update(self.curobo_ignored_substring, robot.articulation_root_path)
                robot.curobo_robot_controller.attach_obj(ee, sim_js, robot.dof_names)
                robot.curobo_pick_place_task.get_place_position(observations)

        else:  # target position has been set
            sim_js = robot.get_joints_state()
            art_action = robot.curobo_robot_controller.forward(ee, sim_js, robot.dof_names)
            if art_action is not None:
                robot.get_articulation_controller().apply_action(art_action)
                # for _ in range(2):
                #    self.omni_world.step(render=False)

    def init_panda_picking_task(self, panda: Panda):
        panda_picking_task = panda.create_picking_task(
            task_name=self.get_robot_task_name(IsaacTaskId.PICK_PLACE_PANDA, panda)
        )
        # This invokes [self.omni_world.reset()] -> task's [set_up_scene()]
        self.add_robot_task(panda_picking_task)

    def step_panda_picking(self, panda: Panda):
        if self.omni_world.is_playing():
            if self.omni_world.current_time_step_index == 0:
                #NOTE: This invokes current tasks' set_up_scene()
                self.omni_world.reset()
            panda.fab_picking_task.step()

    def spawn_camera(self, prim_path: str, position: np.array, orientation: np.array,
                     add_motion_vector: bool = False):
        """
        Spawn, Initialize and calibrate camera sensor
        Ref: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html
        :param camera: camera to be initialized
        :param add_motion_vector: whether to add a motion vector to camera frame
        """
        #camera_prim = self.omni_stage.DefinePrim(prim_path, "Xform")
        camera = Camera(
            prim_path=prim_path,
            position=position,
            frequency=20,
            resolution=(256, 256),
            orientation=orientation if orientation.shape[0] == 4 else
            rotation_utils.euler_angles_to_quats(orientation, degrees=True),
        )
        camera.set_world_pose(position, orientation, camera_axes="usd")

        # Init camera
        camera.initialize()
        if add_motion_vector:
            camera.add_motion_vectors_to_frame()
        camera.add_distance_to_image_plane_to_frame()
        camera.add_pointcloud_to_frame()
        camera.add_occlusion_to_frame()

        # Calibrate camera
        # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
        width, height = 1920, 1200
        camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

        # Pixel size in microns, aperture and focus distance from the camera sensor specification
        # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
        pixel_size = 3 * 1e-3  # in mm, 3 microns is a common pixel size for high resolution cameras
        f_stop = 1.8  # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        focus_distance = 0.6  # in meters, the distance from the camera to the object plane

        # Calculate the focal length and aperture size from the camera matrix
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix
        horizontal_aperture = pixel_size * width  # The aperture size in mm
        vertical_aperture = pixel_size * height
        focal_length_x = fx * pixel_size
        focal_length_y = fy * pixel_size
        focal_length = (focal_length_x + focal_length_y) / 2  # The focal length in mm

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        camera.set_focal_length(focal_length / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
        camera.set_focus_distance(focus_distance)  # The focus distance in meters
        camera.set_lens_aperture(f_stop * 100.0)  # Convert the f-stop to Isaac Sim units
        camera.set_horizontal_aperture(
            horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
        camera.set_vertical_aperture(vertical_aperture / 10.0)

        camera.set_clipping_range(0.05, 1.0e5)

        ######################################
        '''
        from isaacsim.core.api.objects import DynamicCuboid
        cube_2 = self.omni_world.scene.add(
            DynamicCuboid(
                prim_path="/new_cube_2",
                name="cube_1",
                position=np.array([5.0, 3, 1.0]),
                scale=np.array([0.6, 0.5, 0.2]),
                size=1.0,
                color=np.array([255, 0, 0]),
            )
        )
        add_update_semantics(cube_2.prim, "cube")

        cube_3 = self.omni_world.scene.add(
            DynamicCuboid(
                prim_path="/new_cube_3",
                name="cube_2",
                position=np.array([-5, 1, 3.0]),
                scale=np.array([0.1, 0.1, 0.1]),
                size=1.0,
                color=np.array([0, 0, 255]),
                linear_velocity=np.array([0, 0, 0.4]),
            )
        )
        add_update_semantics(cube_3.prim, "cube")
        points_2d = self.camera.get_image_coords_from_world_points(
            np.array([cube_3.get_world_pose()[0], cube_2.get_world_pose()[0]])
        )
        points_3d = self.camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
        print(points_2d)
        print(points_3d)
        rgba = self.camera.get_rgba()
        if rgba:
            plt.imshow(self.camera.get_rgba()[:, :, :3])
            plt.show()
        print(self.camera.get_current_frame()["motion_vectors"])
        '''
        return camera

    def generate_occupancy(self, cell_size, start_location, lower_bound, upper_bound):
        update_location(self.occupancy_map_interface, start_location, lower_bound, upper_bound)
        self.occupancy_map_interface.set_cell_size(cell_size)
        self.occupancy_map_interface.generate()
        self.occupancy_map_interface.get_occupied_positions()
        top_left, top_right, bottom_left, bottom_right, image_coords = compute_coordinates(self.occupancy_map_interface,
                                                                                           cell_size)
        min_b = self.occupancy_map_interface.get_min_bound()
        max_b = self.occupancy_map_interface.get_max_bound()
        return generate_image(self.occupancy_map_generator, [0, 0, 0, 255],
                              [127, 127, 127, 255], [255, 255, 255, 255])
