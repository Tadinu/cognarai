import os
from typing import Optional, Sequence, Union

import numpy as np

# Cognarai
from .ur5 import UR5
from .isaac_common import *


class UR5E_2F_140(UR5):
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
        if end_effector_prim_name is None:
            end_effector_prim_name = "grasp_frame"
        super().__init__(robot_model_name=robot_model_name,
                         description_path=description_path,
                         prim_path=prim_path, name=name, articulation_root_path=articulation_root_path,
                         end_effector_prim_name=end_effector_prim_name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         attach_extra_gripper=False,  # Gripper is already built-in, no need for manual attachment
                         gripper_open_position=gripper_open_position,
                         gripper_closed_position=gripper_closed_position,
                         gripper_fingers_deltas=gripper_fingers_deltas,
                         visible=visible)
        self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                    "ur5e_robotiq_2f_140", "rmpflow", "ur5e_robotiq_2f_140.yaml")
        self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                  "ur5e_robotiq_2f_140", "ur5e_robotiq_2f_140.urdf")
