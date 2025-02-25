from typing_extensions import List, Optional, Union, TYPE_CHECKING

# Omniverse
import numpy as np
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.articulations.articulation_gripper import ArticulationGripper
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from isaacsim.robot.manipulators.controllers.stacking_controller import StackingController as ManipulatorStackingController

# Cognarai
from .pick_place_controller import PickPlaceController


class StackingController(ManipulatorStackingController):
    """[summary]

    Args:
        name (str): [description]
        gripper (object): [description]
        robot_articulation(Articulation): [description]
        picking_order_cube_names (List[str]): [description]
        robot_observation_name (str): [description]
    """

    def __init__(
        self,
        name: str,
        gripper: Union[Gripper, ArticulationGripper],
        robot_articulation: Articulation,
        picking_order_cube_names: List[str],
        robot_observation_name: str
    ) -> None:
        from .omni_robot import OmniRobot
        self.robot: OmniRobot = robot_articulation if isinstance(robot_articulation, OmniRobot) else None
        assert self.robot, f"Robot {robot_articulation.name} is expected to be an instance of {OmniRobot}"

        super().__init__(
            name=name,
            pick_place_controller=PickPlaceController(
                name=f"{name}_pick_place_controller", gripper=gripper, robot_articulation=robot_articulation
            ),
            picking_order_cube_names=picking_order_cube_names,
            robot_observation_name=robot_observation_name,
        )

    def forward(
        self,
        observations: dict,
        end_effector_orientation: Optional[np.ndarray] = None,
        end_effector_offset: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            observations (dict): [description]
            end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        return super().forward(
            observations, end_effector_orientation=end_effector_orientation, end_effector_offset=end_effector_offset
        )
