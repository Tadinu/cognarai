from typing_extensions import List, Optional, TYPE_CHECKING
import numpy as np

# Omniverse/Isaac
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.controllers import PickPlaceController as ManipulatorPickPlaceController
from isaacsim.robot.manipulators.grippers.gripper import Gripper


class PickPlaceController(ManipulatorPickPlaceController):
    """[summary]

    Args:
        name (str): [description]
        gripper (Gripper): [description]
        robot_articulation(Articulation): [description]
        events_dt (Optional[List[float]], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        gripper: Gripper,
        robot_articulation: Articulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        from .omni_robot import OmniRobot
        self.robot: OmniRobot = robot_articulation if isinstance(robot_articulation, OmniRobot) else None
        assert self.robot, f"Robot {robot_articulation.name} is expected to be an instance of {OmniRobot}"

        super().__init__(
            name=name,
            cspace_controller=self.robot.rmp_flow_controller,
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            picking_position (np.ndarray): [description]
            placing_position (np.ndarray): [description]
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        return super().forward(
            picking_position,
            placing_position,
            current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )
