source ACTIVATE_ISAAC_ENV.sh
echo "ISAAC_LAB: $ISAACLAB_PATH"
ISAAC_LAB_EXTS=$ISAACLAB_PATH/source/extensions
export PYTHONPATH=$PYTHONPATH:$ISAACLAB_PATH/source/isaaclab:$ISAACLAB_PATH/source/isaaclab_assets:$ISAACLAB_PATH/source/isaaclab_mimic:$ISAACLAB_PATH/source/isaaclab_rl:$ISAACLAB_PATH/source/isaaclab_tasks:$ISAAC_LAB_EXTS/omni.isaac.lab:$ISAAC_LAB_EXTS/omni.isaac.lab_assets:$ISAAC_LAB_EXTS/omni.isaac.lab_tasks
