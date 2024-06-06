import os
from typing import Optional

import carb
import numpy as np
import omni.isaac.core.objects
import omni.isaac.motion_generation.interface_config_loader as interface_config_loader
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import ArticulationTrajectory
from omni.isaac.motion_generation.lula import RRT
from omni.isaac.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from omni.isaac.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from omni.isaac.motion_generation.path_planning_interface import PathPlanner

# Ref: ~/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.examples/omni/isaac/examples/path_planning/path_planning_controller.py
class PathPlannningController(BaseController):
    def __init__(
        self,
        name: str,
        path_planner_visualizer: PathPlannerVisualizer,
        cspace_trajectory_generator: LulaCSpaceTrajectoryGenerator,
        physics_dt=1 / 60.0,
        rrt_interpolation_max_dist=0.01,
    ):
        BaseController.__init__(self, name)

        robot_articulation = path_planner_visualizer.get_robot_articulation()
        assert robot_articulation, "Path planner visualizer has no robot articulation set up"
        from .omni_robot import OmniRobot
        self.robot: OmniRobot = robot_articulation if isinstance(robot_articulation, OmniRobot) else None
        assert self.robot, f"Robot {robot_articulation.name} is expected to be an instance of {OmniRobot}"

        self._cspace_trajectory_generator = cspace_trajectory_generator
        self._path_planner = path_planner_visualizer.get_path_planner()
        self._path_planner_visualizer = path_planner_visualizer

        self._last_solution = None
        self._action_sequence = None

        self._physics_dt = physics_dt
        self._rrt_interpolation_max_dist = rrt_interpolation_max_dist

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        # This example uses the LulaCSpaceTrajectoryGenerator to convert RRT waypoints to a cspace trajectory.
        # In general this is not theoretically guaranteed to work since the trajectory generator uses spline-based
        # interpolation and RRT only guarantees that the cspace position of the robot can be linearly interpolated between
        # waypoints.  For this example, we verified experimentally that a dense interpolation of cspace waypoints with a maximum
        # l2 norm of .01 between waypoints leads to a good enough approximation of the RRT path by the trajectory generator.

        interpolated_path = self._path_planner_visualizer.interpolate_path(rrt_plan, self._rrt_interpolation_max_dist)
        trajectory = self._cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self.robot, trajectory, self._physics_dt)

        return art_trajectory.get_action_sequence()

    def _make_new_plan(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> None:
        self._path_planner.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._path_planner.update_world()

        path_planner_visualizer = PathPlannerVisualizer(self.robot, self._path_planner)
        active_joints = path_planner_visualizer.get_active_joints_subset()
        if self._last_solution is None:
            start_pos = active_joints.get_joint_positions()
        else:
            start_pos = self._last_solution
        self._rrt_plan = self._path_planner.compute_path(start_pos, np.array([]))

        if self._rrt_plan is None or self._rrt_plan == []:
            carb.log_warn("No plan could be generated to target pose: " + str(target_end_effector_position))
            self._action_sequence = []
            return

        self._action_sequence = self._convert_rrt_plan_to_trajectory(self._rrt_plan)
        self._last_solution = self._action_sequence[-1].joint_positions

    def forward(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> ArticulationAction:
        if self._action_sequence is None:
            # This will only happen the first time the forward function is used
            self._make_new_plan(target_end_effector_position, target_end_effector_orientation)

        if len(self._action_sequence) == 0:
            # The plan is completed; return null action to remain in place
            return ArticulationAction()

        if len(self._action_sequence) == 1:
            final_positions = self._action_sequence[0].joint_positions
            # print("Steady State Error: ", np.linalg.norm(self.robot.get_joint_positions()[:7]-final_positions[:7]))
            return ArticulationAction(
                final_positions, np.zeros_like(final_positions), joint_indices=self._action_sequence[0].joint_indices
            )

        return self._action_sequence.pop(0)

    def add_obstacle(self, obstacle: omni.isaac.core.objects, static: bool = False) -> None:
        self._path_planner.add_obstacle(obstacle, static)

    def remove_obstacle(self, obstacle: omni.isaac.core.objects) -> None:
        self._path_planner.remove_obstacle(obstacle)

    def reset(self) -> None:
        # PathPlannerController will make one plan per reset
        self._path_planner.reset()
        self._action_sequence = None
        self._last_solution = None

    def get_path_planner(self) -> PathPlanner:
        return self._path_planner


class PathRRTController(PathPlannningController):
    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
    ):
        from .omni_robot import OmniRobot
        self.robot: OmniRobot = robot_articulation if isinstance(robot_articulation, OmniRobot) else None
        assert self.robot, f"Robot {robot_articulation.name} is expected to be an instance of {OmniRobot}"

        # Load default RRT config files stored in the omni.isaac.motion_generation extension
        rrt_config = interface_config_loader.load_supported_path_planner_config(self.robot.robot_model_name,
                                                                                "RRT",
                                                                                self.robot.isaac_common.PATH_PLAN_EXTERNAL_CONFIGS_DIRECTORY)

        # Replace the default robot description file with a copy that has inflated collision spheres
        from .isaac_common import IsaacCommon
        rrt_config["robot_description_path"] = self.robot.cspace_description_path
        rrt_config["urdf_path"] = self.robot.lula_description_path
        rrt = RRT(**rrt_config)

        # Create a trajectory generator to convert RRT cspace waypoints to trajectories
        cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(self.robot.cspace_description_path,
                                                                    self.robot.lula_description_path)

        # It is important that the Robot Description File includes optional Jerk and Acceleration limits so that the generated trajectory
        # can be followed closely by the simulated robot Articulation
        assert cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limits()
        assert cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limits()

        visualizer = PathPlannerVisualizer(self.robot, rrt)

        super().__init__(name=name, path_planner_visualizer=visualizer,
                         cspace_trajectory_generator=cspace_trajectory_generator)
