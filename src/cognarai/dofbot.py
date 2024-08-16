import os
from typing import Optional, Sequence, Union

import numpy as np

# Cognarai
from .omni_robot import OmniRobot
from .isaac_common import IsaacCommon


class DofBot(OmniRobot):
    """[summary]

    Args:
        prim_path (str): [description]
        articulation_root_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        description_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
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
        if end_effector_prim_name is None:
            end_effector_prim_name = "link5/Finger_Right_01"
        if gripper_open_position is None:
            gripper_open_position = np.array([0.523599, -0.523599])
        if gripper_closed_position is None:
            gripper_closed_position = np.array([-0.67192185, 0.67192185])
        if gripper_fingers_deltas is None:
            gripper_fingers_deltas = np.array([0.1, -0.1])
        super().__init__(robot_model_name=robot_model_name,
                         description_path=description_path,
                         prim_path=prim_path, name=name, articulation_root_path=articulation_root_path,
                         end_effector_prim_name=end_effector_prim_name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         attach_extra_gripper=False,
                         rmp_policy_name="RMPflow",
                         rmp_policy_with_gripper_name="RMPflowSuction",
                         gripper_open_position=gripper_open_position,
                         gripper_closed_position=gripper_closed_position,
                         gripper_fingers_deltas=gripper_fingers_deltas,
                         visible=visible)
        self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                    "dofbot", "rmpflow", "robot_descriptor.yaml")
        self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                  "dofbot", "arm.urdf")

