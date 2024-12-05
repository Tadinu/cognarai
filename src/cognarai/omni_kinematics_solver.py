from typing_extensions import Optional, Tuple, TYPE_CHECKING

import numpy as np

# Omniverse
import omni.isaac.motion_generation.interface_config_loader as interface_config_loader
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver

# Cognarai
from cognarai.omni_robot import OmniRobot

class OmniKinematicsSolver(ArticulationKinematicsSolver):
    """Kinematics Solver for OmniRobot.  This class loads a LulaKinematicsSovler object

    Args:
        robot (OmniRobot): An initialized robot
        end_effector_frame_name (Optional[str]): The name of the end effector. Defaults to None.
    """

    def __init__(self, robot: OmniRobot, end_effector_frame_name: str) -> None:
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
            robot.isaac_common.get_robot_base_body_model_name(robot.robot_model_name),
            robot.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY
        )
        self._kinematics = LulaKinematicsSolver(**kinematics_config)
        super().__init__(robot, self._kinematics, end_effector_frame_name)

    def fk(self, joint_positions: Optional[np.array] = None, position_only=False) -> Tuple[np.array, np.array]:
        """Doing forward kinematics computation of the robot end effector pose given prior joint positions

        Args:
            joint_positions (np.array) [Optional]: Joint values
            position_only (bool): If True, only the frame positions need to be calculated.  The returned rotation may be left undefined.

        Returns:
            Tuple[np.array,np.array]:
            position: Translation vector describing the translation of the robot end effector relative to the USD global frame (in stage units)
            rotation: (3x3) rotation matrix describing the rotation of the frame relative to the USD stage global frame
        """
        if joint_positions is None:
            return super().compute_end_effector_pose(position_only)

        return self._kinematics_solver.compute_forward_kinematics(
            self._ee_frame, joint_positions, position_only=position_only
        )
