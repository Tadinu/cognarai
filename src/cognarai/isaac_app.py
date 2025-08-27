from typing import Any, Optional

from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import (
    Articulation,
    RigidObject,
    RigidObjectCollection
)
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

print("ISAACLAB_NUCLEUS_DIR:", ISAACLAB_NUCLEUS_DIR)

class IsaacApp:
    def __init__(self, app_launcher:AppLauncher,
                 sim_cfg: Optional[SimulationCfg] = None,
                 interactive_scene_cfg: Optional[InteractiveSceneCfg] = None,
                 args: Optional[Any] = None):
        # Args
        self._args = args

        # App
        self._sim_app = app_launcher.app

        # Sim
        self._sim_cfg = sim_cfg if sim_cfg else SimulationCfg(dt=0.005, device=args.device)
        self._sim_context = SimulationContext(self._sim_cfg)

        # Scene
        # NOTE: Must be after init [SimulationCfg]
        self._interactive_scene = InteractiveScene(interactive_scene_cfg) if interactive_scene_cfg else None

    def _setup_scene(self):
        # Main camera
        self._sim_context.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
        print("[INFO]: Setup complete...")

    @property
    def sim_app(self):
        return self._sim_app

    def run(self):
        self._sim_context.reset()

        """Runs the simulation loop."""
        scene = self._interactive_scene
        rigid_object: RigidObject = scene["object"]
        robot: Articulation = scene["robot"]
        # Define simulation stepping
        sim_dt = self._sim_context.get_physics_dt()
        count = 0
        # Simulation loop
        while self._sim_app.is_running():
            # Reset
            if count % 250 == 0:
                # reset counter
                count = 0
                # reset the scene entities
                # object
                root_state = rigid_object.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                rigid_object.write_root_pose_to_sim(root_state[:, :7])
                rigid_object.write_root_velocity_to_sim(root_state[:, 7:])

                # robot
                # -- root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                print("Robot root state", root_state.shape)
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # -- joint state
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                print("Joint Pos:", joint_pos.shape)
                print("Joint Vel:", joint_vel.shape)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                scene.reset()
                print("[INFO]: Resetting scene state...")

            # Apply action to robot
            robot.set_joint_position_target(robot.data.default_joint_pos)
            # Write data to sim
            scene.write_data_to_sim()
            # Perform step
            self._sim_context.step()
            # Increment counter
            count += 1
            # Update buffers
            scene.update(sim_dt)

    def close(self):
        self._sim_app.close()
