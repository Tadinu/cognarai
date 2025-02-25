from __future__ import annotations
from typing_extensions import List, Optional, Tuple, TYPE_CHECKING
from collections import OrderedDict

# Omniverse
import carb
import numpy as np
from isaacsim.core.prims import XFormPrim, RigidPrim
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCylinder, VisualCylinder
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.api.tasks import BaseTask

# Cognarai
if TYPE_CHECKING:
    from .omni_robot import OmniRobot


class OmniHanoiTowerTask(BaseTask):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "ur10_stacking".
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str, robot: OmniRobot,
        offset: Optional[np.ndarray] = None,
        disk_height: Optional[float] = 0.05
    ) -> None:
        super().__init__(name=name, offset=offset)
        self._robot: OmniRobot = robot
        self._num_of_disks: int = 3
        self._disks: List[XFormPrim] = []
        self._stack_target_position = np.array([-0.3, -0.3, 0.0]) / get_stage_units() + self._offset
        self._disk_height: float = disk_height

    @property
    def robot(self) -> OmniRobot:
        return self._robot

    def set_robot(self) -> OmniRobot:
        return self._robot

    def set_up_scene(self, scene: Scene) -> None:
        """
        Set up a scene, spawning task-specific objects & default env objects like ground plane into it
        :param scene: A scene to spawn objects into
        """
        super().set_up_scene(scene=scene)
        scene.add_default_ground_plane()
        for i in range(self._num_of_disks):
            color = np.random.uniform(size=(3,))
            disk_prim_path = find_unique_string_name(
                initial_name="/World/Disk", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            disk_name = find_unique_string_name(
                initial_name="disk", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            self._disks.append(
                scene.add(
                    DynamicCylinder(
                         name=disk_name,
                         prim_path=disk_prim_path,
                         radius=0.025*self._num_of_disks - 0.02*i,
                         height=self._disk_height,
                         color=color,
                         mass=1.0,
                         position=np.array([0.5, 0.8, 0.3 + i * self._disk_height]) / get_stage_units(),
                         orientation = euler_angles_to_quat(np.array([0, 0, 0])), #np.pi
                    )
                )
            )
            self._task_objects[self._disks[-1].name] = self._disks[-1]

        # NOTE: Don't add self.robot to scene here since it may have been added outside earlier
        self._task_objects[self._robot.name] = self.robot
        self._move_task_objects_to_their_frame()

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        super().pre_step(time_step_index=time_step_index, simulation_time=simulation_time)
        self.robot.update_gripper()

    def post_reset(self) -> None:
        """[summary]"""
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)

    def get_disk_names(self) -> List[str]:
        """[summary]

        Returns:
            List[str]: [description]
        """
        return [self._disks[i].name for i in range(len(self._disks))]

    def set_params(
        self,
        disk_name: Optional[str] = None,
        disk_position: Optional[str] = None,
        disk_orientation: Optional[str] = None,
        stack_target_position: Optional[str] = None,
    ) -> None:
        """[summary]

        Args:
            disk_name (Optional[str], optional): [description]. Defaults to None.
            disk_position (Optional[str], optional): [description]. Defaults to None.
            disk_orientation (Optional[str], optional): [description]. Defaults to None.
            stack_target_position (Optional[str], optional): [description]. Defaults to None.
        """
        if stack_target_position is not None:
            self._stack_target_position = stack_target_position
        if disk_name:
            self._task_objects[disk_name].set_local_pose(position=disk_position, orientation=disk_orientation)

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["stack_target_position"] = {"value": self._stack_target_position, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_positions, _ = self._robot.end_effector.get_local_poses()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_positions[0],
            }
        }
        for i in range(self._num_of_disks):
            disk_position, disk_orientation = self._disks[i].get_local_pose()
            observations[self._disks[i].name] = {
                "position": disk_position,
                "orientation": disk_orientation,
                "target_position": np.array(
                    [
                        self._stack_target_position[0],
                        self._stack_target_position[1],
                        (self._disk_height * i) + self._disk_height / 2.0,
                    ]
                ),
            }
        return observations
