from __future__ import annotations
import logging

import pathlib
from typing_extensions import List, Optional, Sequence, Union, Tuple

# Omniverse
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.articulations import Articulation, ArticulationGripper
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.robot_assembler import AssembledRobot
import omni.isaac.core.utils.numpy.rotations as rotation_utils
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper

# Isaac-interface
from .isaac_common import IsaacCommon
from .path_planning_controller import PathRRTController
from .pick_place_controller import PickPlaceController
from .stacking_controller import StackingController
from .omni_robot_task import OmniStackTask, OmniTargetFollowTask, OmniPathPlanningTask
from .rmpflow_controller import RMPFlowController
from .omni_robot_task import OmniStackTask, OmniTargetFollowTask


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

        # 1- Robot prim from [usd_path] if not existing
        if not is_prim_path_valid(prim_path):
            robot_prim = add_reference_to_stage(
                usd_path=self.isaac_common.get_robot_default_full_usd_path(robot_model_name),
                prim_path=prim_path)
            assert robot_prim
        super().__init__(prim_path=articulation_root_path, name=name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         visible=visible)
        self.articulation_root_path: str = articulation_root_path
        self.description_path: str = description_path  # udrf, mjcf
        self.lula_description_path: str = ""  # lula urdf, mjcf
        self.cspace_description_path: str = ""
        self.assembled_body: Optional[AssembledRobot] = assembled_body

        # 2- EE
        self.end_effector_prim_path = end_effector_prim_name if end_effector_prim_name and is_prim_path_valid(end_effector_prim_name) \
            else None
        self._end_effector: RigidPrim = RigidPrim(prim_path=self.end_effector_prim_path,
                                                  name=f"{self.name}_end_effector") if self.end_effector_prim_path else None

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
            gripper_usd_path = self.isaac_common.get_robot_default_full_usd_path(gripper_model_name)
            if gripper_usd_path:
                from .isaac_common import SHORT_SUCTION_GRIPPER_MODEL
                if gripper_model_name == SHORT_SUCTION_GRIPPER_MODEL:
                    #NOTE:If attaching to [end_effector_prim_path], gripper usd should not already have articulation configured
                    add_reference_to_stage(usd_path=gripper_usd_path, prim_path=self.end_effector_prim_path)
            elif not has_builtin_gripper:
                logging.error(f"[{robot_model_name}:{self.robot_unique_name}] has no gripper model [{gripper_model_name}]'s usd configured")

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
                    f"An articulation should have been added at {self.gripper_articulation_root_path}"
                self._gripper = gripper_class(
                    gripper_dof_names=self.isaac_common.get_gripper_finger_joint_names(robot_model_name),
                    gripper_open_position=gripper_open_position,
                    gripper_closed_position=gripper_closed_position
                )
            else:
                logging.warning(f"{self.robot_unique_name}[{self.robot_model_name}] - {gripper_model_name}: "
                                f"Invalid gripper class {gripper_class}")

        # 4- Camera
        self.camera = self.isaac.spawn_camera(f"{self.end_effector_prim_path}/Camera",
                                              position=np.array([0.05, 0.0, 0.0]),
                                              orientation=rotation_utils.euler_angles_to_quats(np.array([0, 180, 0]),
                                                                                               degrees=True),
                                              add_motion_vector=False) if self.end_effector_prim_path else None

        # 5- RMP
        self.rmp_target_name: str = ""
        self.rmp_policy_name: str = rmp_policy_name
        self.rmp_policy_with_gripper_name: str = rmp_policy_with_gripper_name
        self.rmp_flow_controller: Optional[RMPFlowController] = None

        # 6- Path planning
        self.path_plan_target_name: str = ""
        self.path_rrt_controller: Optional[BaseController] = None

        # 7- Controllers
        self.task_controller: Optional[BaseController] = None
        self.curobo_robot_controller: Optional[BaseController] = None

    def initialize(self, physics_sim_view=None) -> None:
        """Auto-invoked by world.reset()"""
        print(f"Initialize [{self.robot_model_name}]: Robot[{self.robot_unique_name}] - gripper[{self._gripper}]")
        super().initialize(physics_sim_view)

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

        # Create default RMP-flow controller
        if self.isaac_common.is_supported_articulation_model(self.robot_model_name):
            self.rmp_flow_controller = RMPFlowController(name=f"{self.name}_rmp_flow_controller",
                                                         robot_articulation=self,
                                                         attach_extra_gripper=True if self._gripper else False)
            #self.path_rrt_controller = PathRRTController(name=f"{self.name}_path_rrt_controller",
            #                                              robot_articulation=self)

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

    def create_target_following_task(self, task_name: str,
                                     target_name: Optional[str] = None,
                                     target_prim_path: Optional[str] = None,
                                     target_position: Optional[np.ndarray] = None,
                                     target_orientation: Optional[np.ndarray] = None,
                                     offset: Optional[np.ndarray] = None) -> OmniTargetFollowTask:
        """
        Args:
            task_name (str, optional): [description].
            target_prim_path (Optional[str], optional): [description]. Defaults to None.
            target_name (Optional[str], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        # NOTE: This does not create task params yet, refer to FollowTarget class, which will be done in set_up_scene()
        return OmniTargetFollowTask(robot=self,
                                    name=task_name,
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
        # NOTE: This does not create task params yet, refer to FollowTarget class, which will be done in set_up_scene()
        return OmniPathPlanningTask(robot=self,
                                    name=task_name,
                                    target_name=target_name,
                                    target_prim_path=target_prim_path,
                                    target_position=target_position,
                                    target_orientation=target_orientation,
                                    offset=offset)

    def create_stacking_task(self, task_name: str,
                             target_position: Optional[np.ndarray] = None,
                             cube_size: Optional[np.ndarray] = None,
                             offset: Optional[np.ndarray] = None, ) -> OmniStackTask:
        return OmniStackTask(robot=self,
                             name=task_name,
                             target_position=target_position,
                             cube_size=cube_size,
                             offset=offset)
