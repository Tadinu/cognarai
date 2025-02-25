from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from pathlib import Path
from typing_extensions import List, Optional, Sequence, Union, Tuple, TYPE_CHECKING

# Omniverse
import numpy as np
from isaacsim.core.prims import XFormPrim, RigidPrim
if TYPE_CHECKING:
    import isaacsim.core.api.objects
from isaacsim.core.api.objects import DynamicCapsule
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import Articulation
from isaacsim.core.api.articulations.articulation_gripper import ArticulationGripper
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
#from isaacsim.robot_setup.assembler import AssembledRobot
import isaacsim.core.utils.numpy.rotations as rotation_utils
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver

# Cognarai
from cognarai.isaac_common import IsaacCommon
from cognarai.path_planning_controller import PathRRTController
from cognarai.pick_place_controller import PickPlaceController
from cognarai.stacking_controller import StackingController
from cognarai.rmpflow_controller import RMPFlowController
from cognarai.omni_robot_task import (OmniSimpleStackingTask, OmniTargetFollowingTask, OmniTargetFollowingType,
                                      OmniPathPlanningTask)
from cognarai.omni_robot_hanoi_tower_task import OmniHanoiTowerTask


class OmniRobot(Robot):
    """[summary]

    Args:
        robot_model (str): robot model name, case-sensitive. eg: "ur5", "ur5e"
        prim_path (str): [description]
        articulation_root_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        description_path (Optional[str], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
    """

    def __init__(
            self,
            robot_model_name: str,
            description_path: str,
            prim_path: str,
            name: str,
            articulation_root_path: str,
            gripper_articulation_root_path: Optional[str] = None,
            end_effector_prim_name: Optional[str] = None,
            position: Optional[Sequence[float]] = None,
            translation: Optional[Sequence[float]] = None,
            orientation: Optional[Sequence[float]] = None,
            scale: Optional[Sequence[float]] = None,
            attach_extra_gripper: bool = False,
            gripper_open_position: Optional[np.ndarray] = None,
            gripper_closed_position: Optional[np.ndarray] = None,
            gripper_fingers_deltas: Optional[np.ndarray] = None,
            assembled_body: Optional[AssembledRobot] = None,
            rmp_policy_name: Optional[str] = None,
            rmp_policy_with_gripper_name: Optional[str] = None,
            visible: bool = True
    ) -> None:
        """
        :param gripper_open_position: gripper joint positions when opened
        :param gripper_closed_position: gripper joint positions when closed, indexed same as gripper_open_position
        :param gripper_fingers_deltas: deltas to apply for gripper joint positions when opened or closed, indexed same as gripper_open_position
        """
        from .isaac import Isaac
        self.isaac_common = IsaacCommon()
        self.isaac = Isaac()
        self.robot_model_name: str = robot_model_name
        self.robot_unique_name: str = name
        self.robot_prim_path: str = prim_path
        self.active_dof_num: int = 0
        self.ik_solver: Optional[ArticulationKinematicsSolver] = None
        self.physics_sim_view_inited: bool = False

        # 1- Robot prim from [usd_path] if not existing
        if not is_prim_path_valid(prim_path):
            robot_prim = add_reference_to_stage(
                usd_path=self.isaac_common.get_entity_default_full_usd_path(robot_model_name),
                prim_path=prim_path)
            assert robot_prim
        super().__init__(prim_path=articulation_root_path, name=name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         visible=visible)
        XFormPrim(prim_path).set_world_poses(positions=np.array(position).reshape(1, 3),
                                             orientations=np.array(orientation).reshape(1, 4))
        self.articulation_root_path: str = articulation_root_path
        self.description_path: str = description_path  # udrf, mjcf
        self.lula_description_path: str = ""  # lula urdf, mjcf
        self.cspace_description_path: str = ""
        self.assembled_body: Optional[AssembledRobot] = assembled_body

        # 2.1- EE
        self.end_effector_prim_path = end_effector_prim_name if end_effector_prim_name and is_prim_path_valid(end_effector_prim_name) \
            else None
        self._end_effector: RigidPrim = RigidPrim(prim_paths_expr=self.end_effector_prim_path,
                                                  name=f"{self.name}_end_effector") if self.end_effector_prim_path else None

        # 2.2- IK
        from cognarai.omni_kinematics_solver import OmniKinematicsSolver
        self.ik_solver = OmniKinematicsSolver(self, Path(self.end_effector_prim_path).stem) \
                            if self.end_effector_prim_path else None

        # 3- Gripper
        self.gripper_articulation_root_path: Optional[str] = gripper_articulation_root_path
        self.gripper_articulation: Optional[OmniRobot] = None
        self._gripper: Optional[Union[Gripper, ArticulationGripper]] = None
        self.attach_extra_gripper: bool = attach_extra_gripper
        stage_unit = get_stage_units()
        if gripper_open_position is None:
            gripper_open_position = np.array([0.05, 0.05]) / stage_unit
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.0, 0.0])
        if gripper_fingers_deltas is None:
            gripper_fingers_deltas = np.array([0.05, 0.05]) / stage_unit

        # 3.1- Attach gripper from [gripper_usd_path] to [self.end_effector_prim_path]
        has_builtin_gripper = self.isaac_common.has_builtin_gripper_model(robot_model_name)
        if attach_extra_gripper:
            gripper_model_name = self.isaac_common.get_robot_gripper_model_name(robot_model_name)
            gripper_usd_path = self.isaac_common.get_entity_default_full_usd_path(gripper_model_name)
            if gripper_usd_path:
                from .isaac_common import LONG_SUCTION_GRIPPER_MODEL, SHORT_SUCTION_GRIPPER_MODEL
                if gripper_model_name == LONG_SUCTION_GRIPPER_MODEL or gripper_model_name == SHORT_SUCTION_GRIPPER_MODEL:
                    #NOTE:If attaching to [end_effector_prim_path], gripper usd should not already have articulation configured
                    add_reference_to_stage(usd_path=gripper_usd_path, prim_path=self.end_effector_prim_path)
            elif not has_builtin_gripper:
                logger.error(f"[{robot_model_name}:{self.robot_unique_name}] has no gripper model [{gripper_model_name}]'s usd configured")

        # 3.2- Instantiate Gripper from class (SurfaceGripper, ParallelGripper, ArticulationGripper, etc.)
        if attach_extra_gripper or (has_builtin_gripper and not self.assembled_body):
            gripper_class = self.isaac_common.get_gripper_class(self.robot_model_name)
            if not gripper_class:
                pass
            elif issubclass(gripper_class, SurfaceGripper):
                self._gripper = gripper_class(
                    end_effector_prim_path=self.end_effector_prim_path,
                    translate=0.1611, direction="x"
                )
            elif issubclass(gripper_class, ParallelGripper):
                self._gripper = gripper_class(
                    end_effector_prim_path=self.end_effector_prim_path,
                    joint_prim_names=self.isaac_common.get_gripper_finger_joint_names(robot_model_name),
                    joint_opened_positions=gripper_open_position,
                    joint_closed_positions=gripper_closed_position,
                    action_deltas=gripper_fingers_deltas
                )
            elif issubclass(gripper_class, ArticulationGripper):
                assert self.isaac.get_robot_articulation_info(self.gripper_articulation_root_path), \
                    f"No dynamic control interface configured for articulation: {self.gripper_articulation_root_path}"
                self._gripper = gripper_class(
                    gripper_dof_names=self.isaac_common.get_gripper_finger_joint_names(robot_model_name),
                    gripper_open_position=gripper_open_position,
                    gripper_closed_position=gripper_closed_position
                )
            else:
                logger.warning(f"{self.robot_unique_name}[{self.robot_model_name}] - {gripper_model_name}: "
                               f"Invalid gripper class {gripper_class}")

        # 4- Camera
        self.camera = self.isaac.spawn_camera(f"{self.end_effector_prim_path}/Camera",
                                              position=np.array([0.05, 0.0, 0.0]),
                                              orientation=rotation_utils.euler_angles_to_quats(np.array([0, 180, 0]),
                                                                                               degrees=True),
                                              add_motion_vector=False) if self.end_effector_prim_path else None

        # 5- RMP
        self.rmp_policy_name: str = rmp_policy_name
        self.rmp_policy_with_gripper_name: str = rmp_policy_with_gripper_name
        self.rmp_flow_controller: Optional[RMPFlowController] = None

        # 6- Path planning
        self.path_rrt_controller: Optional[PathRRTController] = None

        # 7- Controllers
        self.task_controller: Optional[BaseController] = None
        self.curobo_robot_controller: Optional[BaseController] = None

    def initialize(self, physics_sim_view=None) -> None:
        """Auto-invoked by world.reset()"""
        logger.info(f"Initialize [{self.robot_model_name}]: Robot[{self.robot_unique_name}] - gripper[{self._gripper}]")
        super().initialize(physics_sim_view)
        self.physics_sim_view_inited = True

        # Verify active dofs
        assert self.active_dof_num <= self.num_dof, (f"Robot[{self.robot_unique_name}] has "
                                                     f"active-dof-num: {self.active_dof_num} > full-dof-num: {self.num_dof}")

        # Init EE
        if self._end_effector:
            #self.disable_gravity()
            self._end_effector.initialize(physics_sim_view)
            #self.enable_gravity()

        # Init gripper surrogate
        if self._gripper:
            assert not self.gripper_articulation
            # NOTE: [Gripper] is an abstract class, so not handled here
            if isinstance(self._gripper, SurfaceGripper):
                self._gripper.initialize(
                    physics_sim_view=physics_sim_view,
                    articulation_num_dofs=self.num_dof
                )
            elif isinstance(self._gripper, ParallelGripper):
                self._gripper.initialize(
                    physics_sim_view=physics_sim_view,
                    articulation_apply_action_func=self.apply_action,
                    get_joint_positions_func=self.get_joint_positions,
                    set_joint_positions_func=self.set_joint_positions,
                    dof_names=self.dof_names
                )
            elif isinstance(self._gripper, ArticulationGripper):
                self._gripper.initialize(self.gripper_articulation_root_path,
                                         self.get_articulation_controller())

        # Create default RMP-flow & Path-RRT controllers
        if self.isaac_common.is_supported_articulation_model(self.robot_model_name):
            self.rmp_flow_controller = RMPFlowController(name=f"{self.name}_rmp_flow_controller",
                                                         robot_articulation=self,
                                                         attach_extra_gripper=True if self._gripper else False)
            self.path_rrt_controller = PathRRTController(name=f"{self.name}_path_rrt_controller",
                                                         robot_articulation=self)

        # Register obstacles
        for _, objects in self.isaac.objects.items():
            for obj in objects:
                self.register_obstacle_prim(obj)

    def update(self) -> None:
        self.rmp_update()

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        if self._end_effector:
            self._end_effector.post_reset()
        if self._gripper and isinstance(self._gripper, Gripper):
            self._gripper.post_reset()

    @property
    def pick_place_controller(self) -> Optional[PickPlaceController]:
        return self.task_controller._pick_place_controller if isinstance(self.task_controller, StackingController) else None

    @property
    def end_effector(self) -> Optional[RigidPrim]:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    def fk(self, joint_positions: Optional[np.array] = None, position_only=False) -> Optional[Tuple[np.array, np.array]]:
        if self.ik_solver:
            return self.ik_solver.fk(joint_positions, position_only)
        return None

    def follow_target(self, target_name: str):
        if not target_name:
            logger.error(f"[{self.robot_unique_name}]-follow_target(): target_name is empty")
            return False
        if not self.ik_solver:
            logger.error(f"[{self.robot_unique_name}]-follow_target(): ik_solver is None")
            return False
        return True

    @property
    def gripper(self) -> Optional[Union[Gripper, ArticulationGripper]]:
        """[summary]

        Returns:
            Union[Gripper, ArticulationGripper]: [description]
        """
        return self._gripper

    def update_gripper(self):
        if self._gripper and isinstance(self._gripper, SurfaceGripper):
            self._gripper.update()

    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return np.zeros, np.zeros

    def rmp_update(self):
        if self.rmp_flow_controller:
            # https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_motion_generation_rmpflow.html
            rmpflow = self.rmp_flow_controller.rmp_flow
            # Track any movements of the sphere obstacle
            rmpflow.update_world()

            #Track any movements of the robot base
            robot_base_translation, robot_base_orientation = self.get_world_pose()
            rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)

    def register_obstacle(self, obstacle: isaacsim.core.api.objects):
        if self.rmp_flow_controller:
            self.rmp_flow_controller.add_obstacle(obstacle)  # !NOTE: Isaac still does not have this implemented yet!
            if isinstance(obstacle, DynamicCapsule):
                self.rmp_flow_controller.rmp_flow.add_capsule(obstacle) # Temp usage before rmp_flow.add_obstacle() is ready!
                print(self.name, "-RMP FLOW CONTROLLER: ADD CAPSULE", obstacle.name)
        if self.path_rrt_controller:
            self.path_rrt_controller.add_obstacle(obstacle)
            print(self.name, "-PATH RRT CONTROLLER ADD OBSTACLE", obstacle.name)

    def register_obstacle_prim(self, obstacle: XFormPrim):
        capsule = None
        if self.rmp_flow_controller or self.path_rrt_controller:
            # Temp hardcoding wrapping collision obstacle
            positions, orientations = obstacle.get_world_poses()
            capsule = DynamicCapsule(
                prim_path=f"{obstacle.prim_paths[0]}/collision",
                position=positions[0],
                orientation=orientations[0],
                radius=0.1,
                height=0.5,
                color=np.array([1.0, 0.0, 0.0]),
                mass=1.0
            )
        self.register_obstacle(capsule)

    def create_target_following_task(self, task_name: str,
                                     following_type: Optional[OmniTargetFollowingType] = OmniTargetFollowingType.RMP,
                                     target_name: Optional[str] = None,
                                     target_prim_path: Optional[str] = None,
                                     target_position: Optional[np.ndarray] = None,
                                     target_orientation: Optional[np.ndarray] = None,
                                     offset: Optional[np.ndarray] = None) -> OmniTargetFollowingTask:
        """
        Args:
            task_name (str, optional): [description].
            following_type (OmniTargetFollowingType, optional): [description].
            target_prim_path (Optional[str], optional): [description]. Defaults to None.
            target_name (Optional[str], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        # NOTE: This does not create task params yet, which will be done in set_up_scene()
        # Refer to OmniTargetFollowingTask class
        return OmniTargetFollowingTask(robot=self,
                                       name=task_name,
                                       following_type=following_type,
                                       target_name=target_name,
                                       target_prim_path=target_prim_path,
                                       target_position=target_position,
                                       target_orientation=target_orientation,
                                       offset=offset)

    def create_path_planning_task(self, task_name: str,
                                  target_name: Optional[str] = None,
                                  target_prim_path: Optional[str] = None,
                                  target_position: Optional[np.ndarray] = None,
                                  target_orientation: Optional[np.ndarray] = None,
                                  offset: Optional[np.ndarray] = None) -> OmniPathPlanningTask:
        """
        Args:
            task_name (str, optional): [description].
            target_prim_path (Optional[str], optional): [description]. Defaults to None.
            target_name (Optional[str], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        # NOTE: This does not create task params yet, which will be done in set_up_scene().
        # Refer to OmniPathPlanningTask class
        return OmniPathPlanningTask(robot=self,
                                    name=task_name,
                                    target_name=target_name,
                                    target_prim_path=target_prim_path,
                                    target_position=target_position,
                                    target_orientation=target_orientation,
                                    offset=offset)

    def create_simple_stacking_task(self, task_name: str,
                                    target_position: Optional[np.ndarray] = None,
                                    cube_size: Optional[np.ndarray] = None,
                                    offset: Optional[np.ndarray] = None, ) -> OmniSimpleStackingTask:
        return OmniSimpleStackingTask(robot=self,
                                      name=task_name,
                                      target_position=target_position,
                                      cube_size=cube_size,
                                      offset=offset)

    def create_hanoi_tower_task(self, task_name: str,
                                offset: Optional[np.ndarray] = None, ) -> OmniHanoiTowerTask:
        return OmniHanoiTowerTask(robot=self,
                                  name=task_name,
                                  offset=offset)
