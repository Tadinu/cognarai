import os
from typing import Optional, Sequence, Union

from pathlib import Path
import numpy as np

# Cognarai
from .omni_robot import OmniRobot
from isaacsim.robot.manipulators.examples.universal_robots import KinematicsSolver as URKinematicSolver
from .isaac_common import *


class UR5(OmniRobot):
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
        attach_extra_gripper: bool = True,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        gripper_fingers_deltas: Optional[np.ndarray] = None,
        visible: bool = True
    ) -> None:
        if end_effector_prim_name is None:
            end_effector_prim_name = "tool0"
        gripper_model_name = IsaacCommon().get_robot_gripper_model_name(robot_model_name)
        super().__init__(robot_model_name=robot_model_name,
                         description_path=description_path,
                         prim_path=prim_path, name=name, articulation_root_path=articulation_root_path,
                         end_effector_prim_name=end_effector_prim_name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         attach_extra_gripper=attach_extra_gripper,
                         gripper_open_position=gripper_open_position,
                         gripper_closed_position=gripper_closed_position,
                         gripper_fingers_deltas=gripper_fingers_deltas,
                         rmp_policy_name="RMPflow",
                         rmp_policy_with_gripper_name="RMPflow",
                         visible=visible)
        self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                    "universal_robots", "ur5", "rmpflow", "ur5_robot_description.yaml")
        self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                  "universal_robots", "ur5", "ur5.urdf")
        self.ik_solver = URKinematicSolver(self, Path(end_effector_prim_name).stem, attach_gripper=True)

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        #self.set_joints_default_state(
        #    positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        #)

    def follow_target(self, target_name: str):
        if not super().follow_target(target_name):
            return
        observations = self.isaac.omni_world.get_observations()[target_name]
        actions, succ = self.ik_solver.compute_inverse_kinematics(
            target_position=observations["position"],
            target_orientation=observations["orientation"],
        )
        if succ:
            self.apply_action(actions)
        else:
            carb.log_warn(f"[{self.robot_unique_name}]: IK did not converge to a solution. No action is being taken.")

