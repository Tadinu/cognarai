import os
from typing import Optional, Sequence, Union

import numpy as np

# Cognarai
from .iiwa import IIWA
from .isaac_common import *


class IIWA_ALLEGRO(IIWA):
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
            end_effector_prim_name = "iiwa7_link_ee"
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
                                                    "iiwa_allegro", "rmpflow", "iiwa_allegro_description.yaml")
        self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                  "iiwa_allegro", "iiwa_allegro.urdf")


    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )
