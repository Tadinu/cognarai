from typing_extensions import TYPE_CHECKING

import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation


class RMPFlowController(mg.MotionPolicyController):
    """[summary]

    Args:
        name (str): [description]
        robot_articulation (Articulation): [description]
        physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
        attach_extra_gripper (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self, name: str, robot_articulation: Articulation,
        physics_dt: float = 1.0 / 60.0,
        attach_extra_gripper: bool = True
    ) -> None:
        from .omni_robot import OmniRobot
        self.robot: OmniRobot = robot_articulation if isinstance(robot_articulation, OmniRobot) else None
        assert self.robot, f"Robot {robot_articulation.name} is expected to be an instance of {OmniRobot}"

        # Create RMP-flow
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
            self.robot.robot_model_name,
            self.robot.rmp_policy_with_gripper_name if attach_extra_gripper else self.robot.rmp_policy_name,
            self.robot.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY
        )
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        # Create robot articulation's RMP
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        # Only now init super()
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

