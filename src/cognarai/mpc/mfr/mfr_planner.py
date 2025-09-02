import zerorpc
import os
import argparse
import logging
import time
logging.getLogger().setLevel(logging.INFO)

from mfr_common import DEFAULT_MFR_TASK_NAME, MFR_HEADLESS

# IsaacApp Launcher
# -> Must be always created first before importing Omniverse/Issac-related below
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="MFR Isaac Planner")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--valve", action="store_true")
parser.add_argument("--screwdriver", action="store_true")
parser.add_argument('--task', type=str, default=DEFAULT_MFR_TASK_NAME, help='task to evaluate')
AppLauncher.add_app_launcher_args(parser) # [parser] is updated here with added IsaacLab-specific arguments
args = parser.parse_args()
args.headless = MFR_HEADLESS
app_launcher = AppLauncher(args)

# !NOTE: All related to Isaac must be imported after [AppLauncher]

#import pytorch_volumetric as pv
import pytorch_kinematics as pk

# torch
import torch
torch.set_printoptions(precision=2, sci_mode=False)

# Cognarai
from cognarai.mpc.mfr.allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC
from cognarai.mpc.mfr.allegro_screwdriver import AllegroScrewdriver
from cognarai.mpc.mfr.allegro_cuboid_turning import AllegroCuboidTurning
from cognarai.mpc.mfr.allegro_cuboid_alignment_w_force import AllegroCuboidAlignment
from cognarai.mpc.mfr.allegro_reorientation import AllegroReorientation
from cognarai.mpc.mfr.allegro_env import AllegroManipEnv, get_task_config, get_env
from cognarai.mpc.mfr.utils.allegro_utils import *
from cognarai.mppi.utils.transport import torch_to_bytes, bytes_to_torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class MFRPlanner(object):

    def __init__(self, env: AllegroManipEnv, task_config, args):
        obj_dof = sum(task_config['obj_dof_code'])
        task_config['obj_dof'] = obj_dof
        task_config['task'] = args.task

        # Env
        self.env = env

        # Set up the kinematic chain
        self.chain = pk.build_chain_from_urdf(open(env.manip_cfg.hand_urdf_path).read())

        # Trials
        self.trial_count = 0
        self.task_config = task_config

        # Init
        self.task_config['goal'] = torch.tensor(self.task_config['goal'], device=self.task_config['device']).float()
        self.env.reset()
        fpath = pathlib.Path(f'{CURRENT_DIR}/data/experiments/{self.task_config["experiment_name"]}/trial_{self.trial_count + 1}')
        pathlib.Path.mkdir(fpath, parents=True, exist_ok=True)
        # set up params
        params = self.task_config.copy()
        params.pop('controllers')
        params.update(self.task_config['controllers'])
        params['chain'] = self.chain.to(device=params['device'])
        object_location = self.env.object_init_pos.to(params['device']).float()  # NOTE: this is true for the tasks we have now. We need to pay attention if the root joint is not the root of the asset
        params['object_location'] = object_location

        # Pregrasp
        self.pregrasp(params, fpath)

    def apply_hand_state(self, dof_state_data: bytes, root_state_data: bytes):
        dof_state = bytes_to_torch(dof_state_data)
        size = self.env.hand_dof_pos.shape[0]
        joint_pos = dof_state[:size,:]
        joint_vel = dof_state[size:,:]
        self.env.hand.write_joint_state_to_sim(joint_pos, joint_vel)
        self.env.hand.write_root_state_to_sim(bytes_to_torch(root_state_data))

    def pregrasp(self, params, fpath):
        self.params = params
        obj_dof = params['obj_dof']
        num_fingers = len(params['fingers'])
        robot_dof = 4 * num_fingers
        if params['object_type'] == 'screwdriver':
            "only turn the screwdriver once"
            obj_joint_dim = 1  # compensate for the screwdriver cap
        else:
            obj_joint_dim = 0

        self.env.reset()
        if params['visualize']:
            self.env.frame_fpath = fpath
            self.env.frame_id = 0
        else:
            self.env.frame_fpath = None
            self.env.frame_id = None
        state = self.env.get_state()
        action_list = []
        start = state[0].to(device=params['device'])

        # setup the pregrasp problem
        pregrasp_flag = False
        task = params['task']
        if task == 'cuboid_turning' or task == 'reorientation' or task == 'cuboid_alignment':
            pregrasp_flag = False
        else:
            pregrasp_flag = True
        if pregrasp_flag:
            print("Pregrasping...")
            pregrasp_succ = False
            while pregrasp_succ == False:
                pregrasp_dx = pregrasp_du = robot_dof
                pregrasp_problem = AllegroContactProblem(
                    dx=pregrasp_dx,
                    du=pregrasp_du,
                    start=start[:pregrasp_dx + obj_dof],
                    goal=None,
                    T=4,
                    chain=params['chain'],
                    device=params['device'],
                    object_asset_pos=self.env.object_pos,
                    object_type=params['object_type'],
                    world_trans=self.env.world_trans,
                    fingers=params['fingers'],
                    obj_dof_code=params['obj_dof_code'],
                    obj_joint_dim=obj_joint_dim,
                    fixed_obj=True,
                )

                pregrasp_planner = PositionControlConstrainedSVGDMPC(pregrasp_problem, params)
                pregrasp_planner.warmup_iters = 50

                start_time = time.time()
                best_traj, _ = pregrasp_planner.step(start[:pregrasp_dx])
                print(f"pregrasp solve time: {time.time() - start_time}")

                if params['visualize_plan']:
                    traj_for_viz = best_traj[:, :pregrasp_problem.dx]
                    tmp = start[pregrasp_dx:pregrasp_dx + obj_dof].unsqueeze(0).repeat(traj_for_viz.shape[0], 1)
                    tmp_2 = torch.zeros((traj_for_viz.shape[0], 1)).to(traj_for_viz.device)  # the top jint
                    traj_for_viz = torch.cat((traj_for_viz, tmp, tmp_2), dim=1)
                    viz_fpath = pathlib.PurePath.joinpath(fpath, "pregrasp")
                    img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                    gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                    pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                    pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                    visualize_trajectory(traj_for_viz, pregrasp_problem.viz_contact_scenes, viz_fpath,
                                         pregrasp_problem.fingers, pregrasp_problem.obj_dof + obj_joint_dim,
                                         arm_dof=0)

                for x in best_traj[:, :pregrasp_dx]:
                    action = x.reshape(-1, pregrasp_dx).to(device=self.env.device)  # move the rest fingers
                    self.env.step(action)
                    action_list.append(action)
                if params['mode'] == 'simulation':
                    pregrasp_succ = self.env.check_validity(self.env.get_state().cpu()[0])
                if pregrasp_succ == False:
                    print("pregrasp failed, replanning")
                    self.env.reset()

    def plan(self, step_env: bool = True):
        params = self.params
        obj_dof = params['obj_dof']
        goal = params['goal'].cpu()
        num_fingers = len(params['fingers'])
        robot_dof = 4 * num_fingers
        if params['object_type'] == 'screwdriver':
            obj_joint_dim = 1  # compensate for the screwdriver cap
        else:
            obj_joint_dim = 0

        state = self.env.get_state()
        start = state[0].to(device=params['device'])
        task = params['task']
        if task == 'screwdriver_turning':
            manipulation_problem = AllegroScrewdriver(
                start=start[:robot_dof + obj_dof],
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=self.env.object_init_pos,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                finger_stiffness=params['kp'],
                arm_stiffness=500,
                world_trans=self.env.world_trans,
                fingers=params['fingers'],
                force_balance=False,
                collision_checking=params['collision_checking'],
                obj_gravity=params['obj_gravity'],
                contact_region=params['contact_region'],
            )
        elif task == 'valve_turning':
            manipulation_problem = AllegroValveTurning(
                start=start,
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                object_asset_pos=self.env.object_init_pos,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                world_trans=self.env.world_trans,
                fingers=params['fingers'],
                obj_dof_code=params['obj_dof_code'],
            )
        elif task == 'cuboid_turning':
            manipulation_problem = AllegroCuboidTurning(
                start=start,
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                object_asset_pos=self.env.object_init_pos,
                world_trans=self.env.world_trans,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                device=params['device'],
                fingers=params['fingers'],
                obj_dof_code=params['obj_dof_code'],
                obj_gravity=params['obj_gravity'],
            )
        elif task == 'cuboid_alignment':
            manipulation_problem = AllegroCuboidAlignment(
                start=start,
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                device=params['device'],
                cuboid_asset_pos=self.env.object_init_pos,
                wall_asset_pos=self.env.wall_pose,
                wall_dims=self.env.wall_dims,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                world_trans=self.env.world_trans,
                fingers=params['fingers'],
                obj_gravity=params['obj_gravity'],
                collision_checking=params['collision_checking'],
            )
        elif task == 'reorientation':
            manipulation_problem = AllegroReorientation(
                start=start,
                goal=params['goal'],
                T=params['T'],
                chain=params['chain'],
                object_asset_pos=self.env.object_init_pos,
                world_trans=self.env.world_trans,
                object_location=params['object_location'],
                object_type=params['object_type'],
                friction_coefficient=params['friction_coefficient'],
                device=params['device'],
                fingers=params['fingers'],
                obj_dof_code=params['obj_dof_code'],
                obj_gravity=params['obj_gravity'],
            )
        else:
            raise ValueError(f'Unknown task: {task}')

        manipulation_planner = PositionControlConstrainedSVGDMPC(manipulation_problem, params)
        actual_trajectory = []
        duration = 0

        action_full = torch.zeros((1, self.env.action_space.shape[1]), device=self.env.device)
        for k in range(params['num_steps']):
            state = self.env.get_state()
            start = state[0, :robot_dof + obj_dof].to(device=params['device'])
            current_theta = state[:, -obj_dof:].detach().cpu().numpy()
            actual_trajectory.append(start.clone())
            start_time = time.time()
            best_traj, trajectories = manipulation_planner.step(start)

            solve_time = time.time() - start_time
            print(f"solve time: {solve_time}")
            if k == 0:
                warmup_time = solve_time
            else:
                duration += solve_time
            planned_theta_traj = best_traj[:, robot_dof: robot_dof + obj_dof].detach().cpu().numpy()
            print(f"current theta: {current_theta}")
            print(f"planned theta: {planned_theta_traj}")

            if params['visualize_plan']:
                traj_for_viz = best_traj[:, :manipulation_problem.dx]
                traj_for_viz = torch.cat((start[:manipulation_problem.dx].unsqueeze(0), traj_for_viz), dim=0)
                if obj_joint_dim > 0:
                    tmp = torch.zeros((traj_for_viz.shape[0], obj_joint_dim),
                                      device=best_traj.device)  # add the joint for the screwdriver cap
                    traj_for_viz = torch.cat((traj_for_viz, tmp), dim=1)

                viz_fpath = pathlib.PurePath.joinpath(self.env.frame_fpath, f"timestep_{k}")
                img_fpath = pathlib.PurePath.joinpath(viz_fpath, 'img')
                gif_fpath = pathlib.PurePath.joinpath(viz_fpath, 'gif')
                pathlib.Path.mkdir(img_fpath, parents=True, exist_ok=True)
                pathlib.Path.mkdir(gif_fpath, parents=True, exist_ok=True)
                visualize_trajectory(traj_for_viz, manipulation_problem.viz_contact_scenes, viz_fpath,
                                     manipulation_problem.fingers, manipulation_problem.obj_dof + obj_joint_dim,
                                     arm_dof=0)

            x = best_traj[0, :manipulation_problem.dx + manipulation_problem.du]
            x = x.reshape(1, manipulation_problem.dx + manipulation_problem.du)
            manipulation_problem._preprocess(best_traj.unsqueeze(0))
            equality_constr_dict = manipulation_problem._con_eq(best_traj.unsqueeze(0), compute_grads=False,
                                                                compute_hess=False, verbose=True)
            inequality_constr_dict = manipulation_problem._con_ineq(best_traj.unsqueeze(0), compute_grads=False,
                                                                    compute_hess=False, verbose=True)
            print("--------------------------------------")

            action = x[:, manipulation_problem.dx:manipulation_problem.dx + manipulation_problem.du].to(device=self.env.device)
            print("planned force")
            print(action[:, robot_dof:].reshape(num_fingers + params['num_env_force'], 3))
            print("delta action")
            print(action[:, :robot_dof].reshape(num_fingers, 4))
            action = action[:, :robot_dof]
            # NOTE: this is required since we define action as delta action
            action = action + start.unsqueeze(0)[:, :robot_dof].to(action.device)
            action_full[:, self.env.actuated_dof_indices] = action
            print(action_full.shape, action_full)
            if step_env:
                print("Planner stepping")
                self.env.step(action_full)
        return action_full

    def plan_remotely(self, dof_state_data: bytes, root_state_data: bytes):
        """
        Remote client procedural call
        """
        self.apply_hand_state(dof_state_data, root_state_data)
        return torch_to_bytes(self.plan(step_env=False))

def main():
    # Task Config
    task_config = get_task_config(args.task)

    # Env
    env = get_env(args.task, task_config)

    # Planner
    planner = MFRPlanner(env, task_config, args)

    # Server
    server = zerorpc.Server(planner)
    server.bind("tcp://0.0.0.0:4242")
    print("MFR Planner server running..")
    server.run()


if __name__ == "__main__":
    main()


