import os
from typing_extensions import List, Optional, Sequence

import numpy as np

# Omniverse/Isaac
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper

# Cognarai
from .omni_robot import OmniRobot
from .isaac_common import *
from .panda_picking_task import PandaPickingTask

class Panda(OmniRobot):
    """[summary]

    Args:
        prim_path (str): [description]
        articulation_root_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        description_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        robot_model_name: str,
        description_path: str,
        prim_path: str,
        name: str,
        articulation_root_path: str,
        end_effector_prim_name: Optional[str] = None,
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        gripper_fingers_deltas: Optional[np.ndarray] = None,
        visible: bool = True
    ) -> None:
        if not end_effector_prim_name:
            end_effector_prim_name = "panda_rightfinger"

        super().__init__(robot_model_name=robot_model_name,
                         name=name,
                         description_path=description_path,
                         prim_path=prim_path, articulation_root_path=articulation_root_path,
                         end_effector_prim_name=end_effector_prim_name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         attach_extra_gripper=False,  # Gripper is already built-in, no need for manual attachment
                         gripper_open_position=gripper_open_position,
                         gripper_closed_position=gripper_closed_position,
                         gripper_fingers_deltas=gripper_fingers_deltas,
                         rmp_policy_name="RMPflow",
                         rmp_policy_with_gripper_name="RMPflow",
                         visible=visible)
        assert self.gripper
        assert isinstance(self.gripper, ParallelGripper)
        self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                    "franka", "rmpflow", "robot_descriptor.yaml")
        self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                  "franka", "lula_franka_gen.urdf")
        self.fab_picking_task = PandaPickingTask

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )

    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return (1e15 * np.ones(9), 1e13 * np.ones(9))

    def create_picking_task(self, task_name: str,
                            offset: Optional[np.ndarray] = None, ) -> PandaPickingTask:
        self.fab_picking_task = PandaPickingTask(robot=self, name=task_name, offset=offset)
        return self.fab_picking_task
