from __future__ import annotations
from typing_extensions import Dict, List, Optional, Tuple, TYPE_CHECKING
import os

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

# Fabrics
from cognarai.fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from cognarai.fabrics.helpers.functions import get_rotation_matrix

# Omniverse
import carb, omni.appwindow
from carb.input import KeyboardInput
import numpy as np
from isaacsim.core.prims import XFormPrim, GeometryPrim
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.api.tasks import BaseTask

from isaacsim.core.api.sensors.base_sensor import BaseSensor
from isaacsim.core.utils.types import ArticulationAction

# Cognarai
from cognarai.omni_robot import OmniRobot

GRIPPER_ACTION = 0
GO_UP = 0

def init_keyboard(keyboard_callback):
    # register input callback (this should be refactord into actions in the future)
    input = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    return input.subscribe_to_keyboard_events(keyboard, keyboard_callback)

def close_keyboard(keyboard_sub):
    input = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    input.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)


def on_keyboard_event(event, *args, **kwargs):
    global GRIPPER_ACTION
    global GO_UP
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.C:
            print("Closing gripper")
            GRIPPER_ACTION = 1
            return False
        elif event.input == carb.input.KeyboardInput.O:
            GRIPPER_ACTION = 0
            print("Opening gripper")
            return False
        elif event.input == carb.input.KeyboardInput.U:
            GO_UP = 1
            print("Homing")
            return False
        elif event.input == carb.input.KeyboardInput.D:
            GO_UP = 0
            print("Start picking")
            return False
    return True

class PandaPickingTask(BaseTask):
    def __init__(
            self,
            name: str, robot: OmniRobot,
            offset: Optional[np.ndarray] = None,
            pick_target_height: Optional[float] = 0.05
    ) -> None:
        super().__init__(name=name, offset=offset)
        self._robot: OmniRobot = robot
        self._robot.active_dof_num = 7
        self._num_of_obstacles: int = 3
        self._obstacles: List[GeometryPrim] = []
        self._place_target_position = np.array([-0.3, -0.3, 0.0]) / get_stage_units() + self._offset
        self._pick_target: XFormPrim = None
        self._pick_target_height: float = pick_target_height
        self._urdf_path = f"{os.path.dirname(os.path.abspath(__file__))}/assets/panda_with_finger.urdf"
        self._sensors: List[BaseSensor] = []
        self._steps_cnt: int = 0
        self._sub_keyboard = init_keyboard(on_keyboard_event)

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
        # Ground
        scene.add_default_ground_plane()

        # Obstacles
        for i in range(self._num_of_obstacles):
            color = np.random.uniform(size=(3,))
            obstacle_prim_path = find_unique_string_name(
                initial_name="/World/Obst", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            obstacle_name = find_unique_string_name(
                initial_name="obst", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            obstacle_radius = 0.02 * self._num_of_obstacles - 0.005 * i
            self._obstacles.append(
                VisualSphere(
                    name=obstacle_name,
                    prim_path=obstacle_prim_path,
                    radius=obstacle_radius,
                    color=color,
                    position=np.array([0.5, 0.8, 0.3 + i * obstacle_radius]) / get_stage_units(),
                    orientation=euler_angles_to_quat(np.array([0, 0, 0])),  # np.pi
                )
            )
            self._task_objects[self._obstacles[-1].name] = self._obstacles[-1]
            scene.add(self._obstacles[-1])

        # Pick Target
        self._pick_target = DynamicCuboid(
            name=find_unique_string_name(
                initial_name="pick_cube", is_unique_fn=lambda x: not self.scene.object_exists(x)
            ),
            position=np.array([0, 0, 0]),
            orientation=euler_angles_to_quat(np.array([0, 0, 0])),
            prim_path=find_unique_string_name(
                initial_name="/World/PickCube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            ),
            scale=np.array([0.0515, 0.0515, 0.0515]) / get_stage_units(),
            size=1.0,
            color=np.random.uniform(size=(3,)),
        )
        self._task_objects[self._pick_target.name] = self._pick_target
        scene.add(self._pick_target)

        # NOTE: Don't add self.robot to scene here since it may have been added outside earlier
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        super().pre_step(time_step_index=time_step_index, simulation_time=simulation_time)
        self._robot.update_gripper()

    def post_reset(self) -> None:
        """[summary]"""
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)

    def get_obstacles_names(self) -> List[str]:
        return [self._obstacles[i].name for i in range(len(self._obstacles))]

    def get_pick_target_name(self) -> str:
        return self._pick_target.name

    def set_params(
            self,
            pick_target_name: Optional[str] = None,
            pick_target_position: Optional[str] = None,
            pick_target_orientation: Optional[str] = None,
            place_target_position: Optional[str] = None,
    ) -> None:
        """[summary]

        Args:
            pick_target_name (Optional[str], optional): [description]. Defaults to None.
            pick_target_position (Optional[str], optional): [description]. Defaults to None.
            pick_target_orientation (Optional[str], optional): [description]. Defaults to None.
            place_target_position (Optional[str], optional): [description]. Defaults to None.
        """
        if place_target_position is not None:
            self._place_target_position = place_target_position
        if pick_target_name:
            self._task_objects[pick_target_name].set_local_pose(position=pick_target_position,
                                                                orientation=pick_target_orientation)

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["place_target_position"] = {"value": self._place_target_position, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> Dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            }
        }

        # Obstacles
        for i in range(self._num_of_obstacles):
            obst_position, obst_orientation = self._obstacles[i].get_local_pose()
            observations[self._obstacles[i].name] = {
                "position": obst_position,
                "orientation": obst_orientation
            }

        # Pick Target
        pick_target_pos, pick_target_rot = self._pick_target.get_local_pose()
        observations[self._pick_target.name] = {
            "position": pick_target_pos,
            "orientation": pick_target_rot,
            "target_position": np.array(
                [
                    self._place_target_position[0],
                    self._place_target_position[1],
                    0.5 * self._pick_target_height,
                ]
            ),
        }
        return observations

    def create_goal(self) -> GoalComposition:
        goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
        whole_position = [0.1, 0.6, 0.8]
        angle = np.pi/2 * 1
        rot_matrix = get_rotation_matrix(angle, axis='y')
        goal_1 = np.array([0.107, 0, 0])
        goal_1 = np.dot(rot_matrix, goal_1)
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "desired_position": whole_position,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 3.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "panda_link7",
                "child_link": "panda_hand",
                "desired_position": goal_1.tolist(),
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal2": {
                "weight": 3.0,
                "is_primary_goal": False,
                "indices": [6],
                "desired_position": [np.pi/4],
                "epsilon": 0.05,
                "type": "staticJointSpaceSubGoal",
            }
        }
        return GoalComposition(name="goal", content_dict=goal_dict)

    def init_planner(self):
        """
        Initializes the fabric planner for the panda robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.
        """
        goal = self.create_goal()
        with open(self._urdf_path, "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="panda_link0",
            end_links=["panda_leftfinger"],
        )
        self._planner = ParameterizedFabricPlanner(
            self._robot.active_dof_num,
            forward_kinematics,
        )
        collision_links = ["panda_hand", "panda_link3", "panda_link4"]
        panda_limits = [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ]
        # The planner hides all the logic behind the function set_components.
        self._planner.set_components(
            collision_links=collision_links,
            goal=goal,
            number_obstacles=0,
            number_dynamic_obstacles=len(collision_links),
            limits=panda_limits,
        )
        self._planner.concretize(mode='vel', time_step=0.01)

    def move_robot(self, action: np.ndarray):
        dof_num = self._robot.active_dof_num
        assert action.size >= dof_num
        self._robot.apply_action(ArticulationAction(joint_velocities=action[0:dof_num],
                                                    joint_indices=[i for i in range(dof_num)]))

    def step(self):
        action = np.zeros(9)
        # FIRST STEP
        if self._steps_cnt == 0:
            self.init_planner()
            self.move_robot(action)
            print("Control the gripper using 'o' and 'c' for opening and closing the gripper")
            print("Press 'u' to go to a home pose after grasping")
            print("Press 'd' to go to the object again.")

        # STEPPING
        angle = np.pi / 2 * 1
        rot_matrix = get_rotation_matrix(angle, axis='y')
        goal_1 = np.array([0.107, 0, 0])
        goal_1 = np.dot(rot_matrix, goal_1)
        dof_num = self._robot.active_dof_num
        joint_indices = [i for i in range(dof_num)]
        q=self._robot.get_joint_positions(joint_indices)
        qdot=self._robot.get_joint_velocities(joint_indices)
        pick_target_pos, pick_target_rot = self._pick_target.get_world_pose()
        pick_target_pos[2] += 0.1
        if GO_UP:
            pick_target_pos = [0.5, 0.3, 0.8]
        action[0:dof_num] = self._planner.compute_action(
            q=q,
            qdot=qdot,
            x_goal_0 = np.array(pick_target_pos),
            weight_goal_0=0.7,
            x_goal_1 = goal_1,
            weight_goal_1=6.0,
            x_goal_2 = np.array([np.pi/4]),
            weight_goal_2=6.0,
            radius_body_panda_link3=np.array([0.02]),
            radius_body_panda_link4=np.array([0.02]),
            radius_body_panda_hand=np.array([0.02]),
        )
        distance_to_full_opening = np.linalg.norm(q[-2:] - np.array([0.04, 0.04]))
        distance_to_full_closing = np.linalg.norm(q[-2:] - np.array([0.00, 0.00]))
        if GRIPPER_ACTION == 0:# and distance_to_full_opening > 0.01:
            action[dof_num:] = np.ones(2) * 0.05
        elif GRIPPER_ACTION == 1:# and distance_to_full_closing > 0.01:
            action[dof_num:] = np.ones(2) * -0.05
        else:
            action[dof_num:] = np.zeros(2)
        self.move_robot(action)

        if self._steps_cnt >= 5000:
            close_keyboard(self._sub_keyboard)
        self._steps_cnt += 1
