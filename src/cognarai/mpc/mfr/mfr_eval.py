import argparse
import logging
from tqdm import tqdm
logging.getLogger().setLevel(logging.INFO)

# IsaacApp Launcher
from cognarai.mpc.mfr.mfr_common import MFR_HEADLESS
assert MFR_HEADLESS == False
from cognarai.mpc.mfr.mfr_planner import args, app_launcher
# !NOTE: All Isaac-related packages must be imported after [AppLauncher]

# Cognarai
from cognarai.isaac_app import IsaacApp
from cognarai.mpc.mfr.mfr_planner import MFRPlanner
from cognarai.mpc.mfr.allegro_env import get_task_config, get_env


def main():
    # Task Config
    task_config = get_task_config(args.task)

    # Env
    env = get_env(args.task, task_config)

    # Planner
    planner = MFRPlanner(env, task_config, args)

    # App
    app = IsaacApp(app_launcher, sim_cfg=env.cfg.sim, args=args)
    app._interactive_scene = env.scene

    # Run planning
    for _ in tqdm(range(task_config['num_trials'])):
        ret = planner.plan()
        print("ACTION:", ret)

    # APP CLOSE
    app.close()

if __name__ == "__main__":
    main()