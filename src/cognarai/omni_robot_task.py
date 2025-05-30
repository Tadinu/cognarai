from __future__ import annotations
from typing_extensions import List, Optional, Tuple, TYPE_CHECKING
from collections import OrderedDict
from enum import Enum

# Omniverse
import carb
import numpy as np
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.tasks import FollowTarget, BaseTask, Stacking

# Cognarai
if TYPE_CHECKING:
    from .omni_robot import OmniRobot


class OmniTargetFollowingType(Enum):
    IK = 1
    RMP = 2

class OmniTargetFollowingTask(FollowTarget):
    def __init__(self, name: str, robot: OmniRobot,
                 following_type: OmniTargetFollowingType = OmniTargetFollowingType.RMP,
                 target_prim_path: Optional[str] = None,
                 target_name: Optional[str] = None,
                 target_position: Optional[np.ndarray] = None,
                 target_orientation: Optional[np.ndarray] = None,
                 offset: Optional[np.ndarray] = None,
                 ) -> None:
        if target_orientation is None:
            target_orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, 0]))
        super().__init__(
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        self._robot = robot
        self._following_type = following_type

    @property
    def robot(self) -> OmniRobot:
        return self._robot

    def set_robot(self) -> OmniRobot:
        return self._robot

    @property
    def following_type(self) -> OmniTargetFollowingType:
        return self._following_type

    def is_using_rmp(self):
        return self._following_type is OmniTargetFollowingType.RMP

    @property
    def target_name(self) -> str:
        return self._target_name

    def set_up_scene(self, scene: Scene) -> None:
        """
        Set up a scene, spawning task-specific objects & default env objects like ground plane into it
        :param scene: A scene to spawn objects into
        """
        # NOTE: Call [BaseTask.set_up_scene()], not super(), to avoid auto-adding robot to scene
        BaseTask.set_up_scene(self, scene=scene)
        scene.add_default_ground_plane()
        if self._target_orientation is None:
            self._target_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
        if self._target_prim_path is None:
            self._target_prim_path = find_unique_string_name(
                initial_name=f"/World/{self.name}/DefaultFollowTarget", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        if self._target_name is None:
            self._target_name = find_unique_string_name(
                initial_name=f"{self.name}_follow_target", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
        self.set_params(
            target_prim_path=self._target_prim_path,
            target_position=self._target_position,
            target_orientation=self._target_orientation,
            target_name=self._target_name,
        )

        # NOTE: Don't add self.robot to scene here (like in super().set_up_scene()) since it may have been added outside earlier
        self._task_objects[self._robot.name] = self.robot
        self._move_task_objects_to_their_frame()


class OmniSimpleStackingTask(Stacking):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "ur10_stacking".
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str, robot: OmniRobot,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        if target_position is None:
            target_position = np.array([0.7, 0.7, 0]) / get_stage_units()
        super().__init__(
            name=name,
            cube_initial_positions=np.array([[0.3, 0.3, 0.3], [0.3, -0.3, 0.3]]) / get_stage_units(),
            cube_initial_orientations=None,
            stack_target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        self._robot = robot

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
        # NOTE: Call BaseTask.set_up_scene(), not super(), to avoid auto-adding robot to scene
        BaseTask.set_up_scene(self, scene=scene)
        scene.add_default_ground_plane()
        for i in range(self._num_of_cubes):
            color = np.random.uniform(size=(3,))
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            self._cubes.append(
                scene.add(
                    DynamicCuboid(
                        name=cube_name,
                        position=self._cube_initial_positions[i],
                        orientation=self._cube_initial_orientations[i],
                        prim_path=cube_prim_path,
                        scale=self._cube_size,
                        size=1.0,
                        color=color,
                    )
                )
            )
            self._task_objects[self._cubes[-1].name] = self._cubes[-1]
        # NOTE: Don't add self.robot to scene here (like in super().set_up_scene()) since it may have been added outside earlier
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

# Ref: <ISAAC_SIM>/exts/omni.isaac.examples/omni/isaac/examples/path_planning/path_planning_task.py
class OmniPathPlanningTask(BaseTask):
    def __init__(
        self,
        name: str,
        robot: OmniRobot,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:

        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = robot
        self._target_name = target_name
        self._target = None
        self._target_prim_path = target_prim_path
        self._target_position = target_position
        self._target_orientation = target_orientation
        self._target_visual_material = None
        self._obstacle_walls = OrderedDict()
        if self._target_position is None:
            self._target_position = np.array([0.8, 0.3, 0.4]) / get_stage_units()
        return

    @property
    def robot(self) -> OmniRobot:
        return self._robot

    def set_robot(self) -> OmniRobot:
        return self._robot

    @property
    def target_name(self) -> str:
        return self._target_name

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)

        # Ground
        scene.add_default_ground_plane()

        # Target
        if self._target_orientation is None:
            self._target_orientation = euler_angles_to_quat(np.array([-np.pi, 0, np.pi]))
        if self._target_prim_path is None:
            self._target_prim_path = find_unique_string_name(
                initial_name="/World/TargetCube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        if self._target_name is None:
            self._target_name = find_unique_string_name(
                initial_name="target", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
        self.set_params(
            target_prim_path=self._target_prim_path,
            target_position=self._target_position,
            target_orientation=self._target_orientation,
            target_name=self._target_name,
        )
        # NOTE: Don't add self.robot to scene here since it may have been added outside earlier
        self._robot = self.set_robot()
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()

    def set_params(
        self,
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            target_prim_path (Optional[str], optional): [description]. Defaults to None.
            target_name (Optional[str], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        if target_prim_path is not None:
            if self._target is not None:
                del self._task_objects[self._target.name]
            if is_prim_path_valid(target_prim_path):
                self._target = self.scene.add(
                    XFormPrim(
                        prim_path=target_prim_path,
                        position=target_position,
                        orientation=target_orientation,
                        name=target_name,
                    )
                )
            else:
                self._target = self.scene.add(
                    VisualCuboid(
                        name=target_name,
                        prim_path=target_prim_path,
                        position=target_position,
                        orientation=target_orientation,
                        color=np.array([1, 0, 0]),
                        size=1.0,
                        scale=np.array([0.03, 0.03, 0.03]) / get_stage_units(),
                    )
                )
            self._task_objects[self._target.name] = self._target
            self._target_visual_material = self._target.get_applied_visual_material()
            if self._target_visual_material is not None:
                if hasattr(self._target_visual_material, "set_color"):
                    self._target_visual_material.set_color(np.array([1, 0, 0]))
        else:
            self._target.set_local_pose(position=target_position, orientation=target_orientation)

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["target_prim_path"] = {"value": self._target.prim_path, "modifiable": True}
        params_representation["target_name"] = {"value": self._target.name, "modifiable": True}
        position, orientation = self._target.get_local_pose()
        params_representation["target_position"] = {"value": position, "modifiable": True}
        params_representation["target_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        target_position, target_orientation = self._target.get_local_pose()
        return {
            self._robot.name: {
                "joint_positions": np.array(joints_state.positions),
                "joint_velocities": np.array(joints_state.velocities),
            },
            self._target.name: {"position": np.array(target_position), "orientation": np.array(target_orientation)},
        }

    def target_reached(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        target_position, _ = self._target.get_world_pose()
        if np.mean(np.abs(np.array(end_effector_position) - np.array(target_position))) < (0.035 / get_stage_units()):
            return True
        else:
            return False

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        if self._target_visual_material is not None:
            if hasattr(self._target_visual_material, "set_color"):
                if self.target_reached():
                    self._target_visual_material.set_color(color=np.array([0, 1.0, 0]))
                else:
                    self._target_visual_material.set_color(color=np.array([1.0, 0, 0]))

        return

    def add_obstacles(self):
        self.add_obstacle()

    def add_obstacle(self, position: np.ndarray = None, orientation=None):
        """[summary]

        Args:
            position (np.ndarray, optional): [description]. Defaults to np.array([0.1, 0.1, 1.0]).
        """
        # TODO: move to task frame if there is one
        cube_prim_path = find_unique_string_name(
            initial_name="/World/WallObstacle", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="wall", is_unique_fn=lambda x: not self.scene.object_exists(x))
        if position is None:
            position = np.array([0.6, 0.1, 0.3]) / get_stage_units()
        if orientation is None:
            orientation = euler_angles_to_quat(np.array([0, 0, np.pi / 3]))
        cube = self.scene.add(
            VisualCuboid(
                name=cube_name,
                position=position + self._offset,
                orientation=orientation,
                prim_path=cube_prim_path,
                size=1.0,
                scale=np.array([0.1, 0.5, 0.6]) / get_stage_units(),
                color=np.array([0, 0, 1.0]),
            )
        )
        self._obstacle_walls[cube.name] = cube

        # Register task's obstacles to robot
        assert self.robot.physics_sim_view_inited, "Robot's physics sim view must be initialized before registering obstacles to it!"
        self.robot.register_obstacle(cube)
        return cube

    def remove_obstacle(self, name: Optional[str] = None) -> None:
        """[summary]

        Args:
            name (Optional[str], optional): [description]. Defaults to None.
        """
        if name is not None:
            self.scene.remove_object(name)
            self._robot.path_rrt_controller.remove_obstacle(self._obstacle_walls[name])
            del self._obstacle_walls[name]
        else:
            obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
            self.scene.remove_object(obstacle_to_delete)
            self._robot.path_rrt_controller.remove_obstacle(self._obstacle_walls[obstacle_to_delete])
            del self._obstacle_walls[obstacle_to_delete]

    def get_obstacles(self) -> OrderedDict[str, VisualCuboid]:
        return self._obstacle_walls

    def get_obstacle_to_delete(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
        return self.scene.get_object(obstacle_to_delete)

    def obstacles_exist(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return len(self._obstacle_walls) > 0

    def cleanup(self) -> None:
        """[summary]"""
        obstacles_to_delete = list(self._obstacle_walls.keys())
        for obstacle_to_delete in obstacles_to_delete:
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]



