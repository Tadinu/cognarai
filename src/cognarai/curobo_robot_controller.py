import os
from typing import Optional

import numpy
import numpy as np

import torch
import carb
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, cuboid
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots import Robot
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import BaseTask, Stacking, PickPlace
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
#from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig

# Cognarai
from .omni_robot import OmniRobot


def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


class CuroboRobotController(BaseController):
    def __init__(
        self,
        robot: OmniRobot,
        world: World,
        task: BaseTask,
        name: str = "",
        constrain_grasp_approach: bool = False,
    ) -> None:
        if not name:
            name = f"{robot.robot_unique_name}_curobo_robot_controller"
        BaseController.__init__(self, name=name)
        self._save_log = False
        self.robot = robot
        self.world = world
        self.task = task
        self._step_idx = 0
        self.constrain_grasp_approach = constrain_grasp_approach
        n_obstacle_cuboids = 20
        n_obstacle_mesh = 2
        # warmup curobo instance
        self.usd_help = UsdHelper()
        self.init_curobo = False
        self.world_file = "collision_table.yml"
        self.tensor_device = TensorDeviceType()
        from .isaac import Isaac
        if robot.robot_model_name not in Isaac().entity_configs:
            assert False, f"[{robot.robot_model_name}] model yaml configs have not been loaded yet!"
        self.robot_cfg = Isaac().entity_configs[robot.robot_model_name]
        #self.robot_cfg = load_yaml(Isaac().config_paths[robot.robot_model_name])["robot_cfg"]
        # self.robot_cfg["kinematics"]["cspace"]["max_acceleration"] = 10.0 # controls how fast robot moves
        #self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        # @self.robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
        #self.robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"

        world_config_yml_path = os.path.join(robot.isaac_common.CUROBO_EXTERNAL_CONFIGS_DIRECTORY, "world", "collision_table.yml")
        print(world_config_yml_path)
        world_cfg_table = WorldConfig.from_dict(load_yaml(world_config_yml_path))
        self.world_cfg_table = world_cfg_table

        world_cfg1 = world_cfg_table.get_mesh_world()
        world_cfg1.mesh[0].pose[2] = -10.5

        self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
        self.cmd_js_names = self.robot_cfg.kinematics.cspace.joint_names

        #self.init_motion_gen(n_obstacle_cuboids, n_obstacle_mesh)
        self.init_mpc(n_obstacle_cuboids, n_obstacle_mesh)

        # Start loading stage
        self.usd_help.load_stage(self.world.stage)
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None

    def init_motion_gen(self, n_obstacle_cuboids, n_obstacle_mesh):
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_device,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            store_ik_debug=self._save_log,
            store_trajopt_debug=self._save_log,
            velocity_scale=0.75,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        ee = motion_gen_config.robot_cfg.kinematics.kinematics_config.ee_links[0]
        self.motion_gen.warmup(ee, parallel_finetune=True)
        pose_metric = None
        if self.constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.1, tstep_fraction=0.8
            )

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=10,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=pose_metric,
        )

    def init_mpc(self, n_obstacle_cuboids, n_obstacle_mesh):
        self.mpc_target_past_pose = None
        self.mpc_cmd_state_full = None
        # Make a target to follow
        self.mpc_target = cuboid.VisualCuboid(
            "/World/target",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )

        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            self_collision_check=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            use_mppi=True,
            use_lbfgs=False,
            use_es=False,
            store_rollouts=True,
            step_dt=0.02,
        )
        self.mpc = MpcSolver(mpc_config)
        retract_cfg = self.mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = self.mpc.rollout_fn.joint_names

        ee = self.robot_cfg.kinematics.kinematics_config.ee_links[0]
        state = self.mpc.rollout_fn.compute_kinematics(ee,
                                                       JointState.from_position(retract_cfg, joint_names=joint_names))
        self.mpc_current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
        retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        goal = Goal(
            current_state=self.mpc_current_state,
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
            goal_pose=retract_pose,
        )

        self.mpc_goal_buffer = self.mpc.setup_solve_single(goal, 1)
        self.mpc.update_goal(self.mpc_goal_buffer)
        mpc_result = self.mpc.step(ee, self.mpc_current_state, max_attempts=2)

    def attach_obj(
        self,
        ee: str,
        sim_js: JointState,
        js_names: list,
    ) -> None:
        cube_name = self.task.get_cube_prim(self.task.target_cube)

        cu_js = JointState(
            position=self.tensor_device.to_device(sim_js.positions),
            velocity=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )

        self.motion_gen.attach_objects_to_robot(ee,
            cu_js,
            [cube_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_device),
        )

    def detach_obj(self) -> None:
        self.motion_gen.detach_object_from_robot()

    def plan(
        self,
        ee: str,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        sim_js: JointState,
        js_names: list,
    ) -> MotionGenResult:
        ik_goal = Pose(
            position=self.tensor_device.to_device(ee_translation_goal),
            quaternion=self.tensor_device.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_device.to_device(sim_js.positions),
            velocity=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        result = self.motion_gen.plan_single(ee, cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        if self._save_log:  # and not result.success.item(): # logging for debugging
            UsdHelper.write_motion_gen_log(
                result,
                {"robot_cfg": self.robot_cfg},
                self.world_cfg,
                cu_js,
                ik_goal,
                join_path("log/usd/", "cube") + "_debug",
                write_ik=False,
                write_trajopt=True,
                visualize_robot_spheres=True,
                link_spheres=self.motion_gen.kinematics.kinematics_config.link_spheres,
                grid_space=2,
                write_robot_usd_path="log/usd/assets",
            )
        return result

    def forward(
        self,
        ee: str,
        sim_js: JointState,
        js_names: list,
    ) -> Optional[ArticulationAction]:
        assert self.task.target_position is not None
        assert self.task.target_cube is not None

        if sim_js is None:
            return None

        if self.cmd_plan is None:
            self.cmd_idx = 0
            self._step_idx = 0
            # Set EE goals
            ee_translation_goal = self.task.target_position
            ee_orientation_goal = np.array([0, 0, -1, 0])
            # compute curobo solution:
            result = self.plan(ee, ee_translation_goal, ee_orientation_goal, sim_js, js_names)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                self.idx_list = [i for i in range(len(self.cmd_js_names))]
                self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_js_names)
            else:
                carb.log_warn("Plan did not converge to a solution.")
                return None
        if self._step_idx % 3 == 0:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() * 0.0,
                joint_indices=self.idx_list,
            )
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
        else:
            art_action = None
        self._step_idx += 1
        return art_action

    def reached_target(self, observations: dict) -> bool:
        curr_ee_position = observations[self.robot.name]["end_effector_position"]
        if np.linalg.norm(
                self.task.target_position - curr_ee_position
        ) < 0.04 and (  # This is half gripper width, curobo succ threshold is 0.5 cm
            self.cmd_plan is None
        ):
            if self.task.cube_in_hand is None:
                print("reached picking target: ", self.task.target_cube)
            else:
                print("reached placing target: ", self.task.target_cube)
            return True
        else:
            return False

    def reset(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # init
        self.update(ignore_substring, robot_prim_path)
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0

    def update(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # print("updating world...")
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()
        # add ground plane as it's not readable:
        obstacles.add_obstacle(self.world_cfg_table.cuboid[0])
        self.motion_gen.update_world(obstacles)
        self.world_cfg = obstacles

    def step_mpc(self):
        from .isaac import Isaac
        omni_world = Isaac().omni_world
        if not self.init_curobo:
            for _ in range(10):
                omni_world.step(render=True)
            self.init_curobo = True
        draw_points(self.mpc.get_visual_rollouts())

        omni_world.step(render=True)
        if not omni_world.is_playing():
            return

        self._step_idx = Isaac().omni_world.current_time_step_index
        if self._step_idx <= 2:
            Isaac().omni_world.reset()
            idx_list = [self.robot.get_dof_index(x) for x in self.cmd_js_names]
            self.robot.set_joint_positions(self.robot_cfg.kinematics.cspace.retract_config.cpu().numpy(), idx_list)

            self.robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        self._step_idx += 1
        if self._step_idx % 1000 == 0:
            print("Updating world")
            obstacles = self.usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                ignore_substring=[
                    self.robot.robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=self.robot.robot_prim_path,
            )
            obstacles.add_obstacle(self.world_cfg_table.cuboid[0])
            self.mpc.world_coll_checker.load_collision_model(obstacles)

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = self.mpc_target.get_world_pose()

        if self.mpc_target_past_pose is None:
            self.mpc_target_past_pose = cube_position + 1.0

        if np.linalg.norm(cube_position - self.mpc_target_past_pose) > 1e-3:
            # Set EE teleop goals, use cube for simple non-vr init:
            ik_goal = Pose(
                position=self.tensor_device.to_device(cube_position),
                quaternion=self.tensor_device.to_device(cube_orientation),
            )
            self.mpc_goal_buffer.goal_pose.copy_(ik_goal)
            self.mpc.update_goal(self.mpc_goal_buffer)
            self.mpc_target_past_pose = cube_position

        # if not changed don't call curobo:

        # get robot current state:
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        sim_js_names = self.robot.dof_names

        cu_js = JointState(
            position=self.tensor_device.to_device(sim_js.positions),
            velocity=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_device.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.mpc.rollout_fn.joint_names)
        if self.mpc_cmd_state_full is None:
            self.mpc_current_state.copy_(cu_js)
        else:
            self.mpc_current_state_partial = self.mpc_cmd_state_full.get_ordered_joint_state(
                self.mpc.rollout_fn.joint_names
            )
            self.mpc_current_state.copy_(self.mpc_current_state_partial)
            self.mpc_current_state.joint_names = self.mpc_current_state_partial.joint_names
            # self.mpc_current_state = self.mpc_current_state.get_ordered_joint_state(self.mpc.rollout_fn.joint_names)
        common_js_names = []
        self.mpc_current_state.copy_(cu_js)

        ee = self.robot_cfg.kinematics.kinematics_config.ee_links[0]
        mpc_result = self.mpc.step(ee, self.mpc_current_state, max_attempts=2)
        # ik_result = ik_solver.solve_single(ee, ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = True  # ik_result.success.item()
        self.mpc_cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in self.mpc_cmd_state_full.joint_names:
                idx_list.append(self.robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = self.mpc_cmd_state_full.get_ordered_joint_state(common_js_names)
        self.mpc_cmd_state_full = cmd_state

        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        # positions_goal = articulation_action.joint_positions
        if self._step_idx % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(3):
                self.robot.get_articulation_controller().apply_action(art_action)
        else:
            carb.log_warn("No action is being taken.")


class CuroboBoxStackTask(Stacking):
    def __init__(
        self,
        robot: OmniRobot,
        name: str = "box_stacking",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        Stacking.__init__(
            self,
            name=name,
            cube_initial_positions=np.array(
                [
                    [0.50, 0.0, 0.1],
                    [0.50, -0.20, 0.1],
                    [0.50, 0.20, 0.1],
                    [0.30, -0.20, 0.1],
                    [0.30, 0.0, 0.1],
                    [0.30, 0.20, 0.1],
                    [0.70, -0.20, 0.1],
                    [0.70, 0.0, 0.1],
                    [0.70, 0.20, 0.1],
                ]
            )
            / get_stage_units(),
            cube_initial_orientations=None,
            stack_target_position=None,
            cube_size=np.array([0.045, 0.045, 0.07]),
            offset=offset,
        )
        self._robot = robot
        self.cube_list = None
        self.target_position = None
        self.target_cube = None
        self.cube_in_hand = None

    def reset(self) -> None:
        self.cube_list = self.get_cube_names()
        self.target_position = None
        self.target_cube = None
        self.cube_in_hand = None

    def post_reset(self) -> None:
        pass

    def set_up_scene(self, scene: Scene) -> None:
        # Don't call super(), which adds `self._robot` again
        BaseTask.set_up_scene(self, scene)
        self.spawn_boxes()

    def spawn_boxes(self):
        for i in range(self._num_of_cubes):
            color = np.random.uniform(size=(3,))
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            cube_name = find_unique_string_name(
                initial_name="cube", is_unique_fn=lambda x: not self._scene.object_exists(x)
            )
            self._cubes.append(
                self._scene.add(
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
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()

    def update_task(self) -> bool:
        # after detaching the cube in hand
        assert self.target_cube is not None
        assert self.cube_in_hand is not None
        self.cube_list.insert(0, self.cube_in_hand)
        self.target_cube = None
        self.target_position = None
        self.cube_in_hand = None
        return len(self.cube_list) <= 1

    def get_cube_prim(self, cube_name: str):
        for i in range(self._num_of_cubes):
            if cube_name == self._cubes[i].name:
                return self._cubes[i].prim_path

    def get_place_position(self, observations: dict) -> None:
        assert self.target_cube is not None
        self.cube_in_hand = self.target_cube
        self.target_cube = self.cube_list[0]
        ee_to_grasped_cube = (
            observations[self._robot.name]["end_effector_position"][2]
            - observations[self.cube_in_hand]["position"][2]
        )
        self.target_position = observations[self.target_cube]["position"] + [
            0,
            0,
            self._cube_size[2] + ee_to_grasped_cube + 0.02,
        ]
        self.cube_list.remove(self.target_cube)

    def get_pick_position(self, observations: dict) -> None:
        assert self.cube_in_hand is None
        self.target_cube = self.cube_list[1]
        self.target_position = observations[self.target_cube]["position"] + [
            0,
            0,
            self._cube_size[2] / 2 + 0.092,
        ]
        self.cube_list.remove(self.target_cube)

    def set_robot(self) -> Robot:
        return self._robot

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose() if self._robot.end_effector \
            else RigidPrim(prim_path=self._robot.end_effector_prim_path).get_local_pose()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions if joints_state else np.zeros(3),
                "end_effector_position": end_effector_position,
            }
        }
        for i in range(self._num_of_cubes):
            cube_position, cube_orientation = self._cubes[i].get_local_pose()
            observations[self._cubes[i].name] = {
                "position": cube_position,
                "orientation": cube_orientation,
                "target_position": np.array(
                    [
                        self._stack_target_position[0],
                        self._stack_target_position[1],
                        (self._cube_size[2] * i) + self._cube_size[2] / 2.0,
                    ]
                ),
            }
        return observations