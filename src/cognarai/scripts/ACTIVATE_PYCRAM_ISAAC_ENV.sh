workon isaac
source ./SOURCE_ALL.sh

#pushd $OMNI_KIT
#source setup_python_env.sh
#popd

pushd $ISAAC_PATH
source setup_python_env.sh
popd

$ISAACSIM_PYTHON -c "import os, torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(os.environ.get('CUDA_PATH')); print(torch.cuda.nccl.version()); \
					 print(torch._C._GLIBCXX_USE_CXX11_ABI)"

# REF: [$ISAACSIM_PATH/setup_conda_env.sh -> setup_python_env.sh]
# DO NOT ADD [$ISAACSIM_PATH/kit/python/lib/python3.10/site-packages], which just confuse notebook server
# CORE EXTS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/exts/omni.isaac.kit:$ISAACSIM_PATH/exts/omni.isaac.core:$ISAACSIM_PATH/exts/omni.isaac.gym:$ISAACSIM_PATH/kit/kernel/py:$ISAACSIM_PATH/kit/plugins/bindings-python:$ISAACSIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAACSIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAACSIM_PATH/kit/exts/omni.kit.pip_archive/pip_prebundle:$ISAACSIM_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAACSIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAACSIM_PATH/exts/omni.pip.compute/pip_prebundle:$ISAACSIM_PATH/exts/omni.pip.cloud/pip_prebundle

# EXTS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/exts/omni.isaac.extension_templates:$ISAACSIM_PATH/exts/omni.isaac.dynamic_control:$ISAACSIM_PATH/exts/omni.isaac.lula:$ISAACSIM_PATH/exts/omni.isaac.manipulators:$ISAACSIM_PATH/exts/omni.isaac.franka:$ISAACSIM_PATH/exts/omni.isaac.universal_robots:$ISAACSIM_PATH/exts/omni.isaac.surface_gripper:$ISAACSIM_PATH/exts/omni.isaac.occupancy_map:$ISAACSIM_PATH/exts/omni.isaac.motion_generation:$ISAACSIM_PATH/exts/omni.isaac.scene_blox:$ISAACSIM_PATH/exts/omni.isaac.examples:$ISAACSIM_PATH/exts/omni.isaac.dynamic_control:$ISAACSIM_PATH/exts/omni.isaac.sensor:$ISAACSIM_PATH/exts/omni.isaac.robot_assembler

# KIT-EXTCORE
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/kit/extscore:$ISAACSIM_PATH/kit/extscore/omni.client:$ISAACSIM_PATH/kit/extscore/omni.assets.plugins:$ISAACSIM_PATH/kit/extscore/omni.kit.async_engine:$ISAACSIM_PATH/kit/extscore/omni.kit.registry.nucleus

# KIT-EXTS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/kit/exts/omni.appwindow:$ISAACSIM_PATH/kit/exts/omni.kit.commands:$ISAACSIM_PATH/kit/exts/omni.kit.actions.core:$ISAACSIM_PATH/kit/exts/omni.kit.actions.window:$ISAACSIM_PATH/kit/exts/omni.timeline:$ISAACSIM_PATH/kit/exts/omni.kit.numpy.common:$ISAACSIM_PATH/kit/exts/omni.kit.material.library:$ISAACSIM_PATH/exts/omni.isaac.ui:$ISAACSIM_PATH/kit/exts/omni.ui:$ISAACSIM_PATH/kit/exts/omni.kit.renderer.imgui:$ISAACSIM_PATH/kit/exts/omni.gpu_foundation:$ISAACSIM_PATH/kit/exts/omni.kit.renderer.core:$ISAACSIM_PATH/kit/exts/omni.ui.scene:$ISAACSIM_PATH/kit/exts/omni.kit.helper.file_utils:$ISAACSIM_PATH/kit/exts/omni.kit.widget.nucleus_connector:$ISAACSIM_PATH/kit/exts/omni.kit.search_core:$ISAACSIM_PATH/exts/omni.isaac.version:$ISAACSIM_PATH/kit/exts/omni.kit.test

# URDF/MJCF IMPORTER
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/exts/omni.isaac.urdf:$ISAACSIM_PATH/extscache/omni.importer.urdf-1.6.1+105.1.lx64.r.cp310:$ISAACSIM_PATH/extscache/omni.importer.mjcf-1.1.0+105.1.lx64.r.cp310

# OMNI.KIt.WINDOW-WIDGET
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/kit/exts/omni.kit.window.file_importer:$ISAACSIM_PATH/kit/exts/omni.kit.window.filepicker:$ISAACSIM_PATH/kit/exts/omni.kit.widget.filebrowser:$ISAACSIM_PATH/kit/exts/omni.kit.window.popup_dialog:$ISAACSIM_PATH/kit/exts/omni.kit.widget.search_delegate:$ISAACSIM_PATH/kit/exts/omni.kit.widget.nucleus_info:$ISAACSIM_PATH/kit/exts/omni.kit.widget.versioning:$ISAACSIM_PATH/kit/exts/omni.kit.widget.browser_bar:$ISAACSIM_PATH/kit/exts/omni.kit.widget.path_field:$ISAACSIM_PATH/kit/exts/omni.kit.notification_manager:$ISAACSIM_PATH/kit/exts/omni.kit.menu.common:$ISAACSIM_PATH/kit/exts/omni.kit.menu.utils:$ISAACSIM_PATH/kit/exts/omni.kit.window.extensions:$ISAACSIM_PATH/kit/exts/omni.kit.widget.graph:$ISAACSIM_PATH/kit/exts/omni.kit.widget.filter:$ISAACSIM_PATH/kit/exts/omni.kit.widget.options_menu

# EXTS PHYSICS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/extsPhysics/omni.physics.tensors-105.1.12-5.1:$ISAACSIM_PATH/extsPhysics/omni.physx-105.1.12-5.1

# TORCH
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/extscache/omni.pip.torch-2_0_1-2.0.2+105.1.lx64/torch-2-0-1

# ROS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/exts/omni.isaac.ros_bridge:$PYTHONPATH:$ISAACSIM_PATH/exts/omni.isaac.ros_bridge/noetic/local/lib/python3.10/dist-packages:$ISAACSIM_PATH/exts/omni.isaac.ros2_bridge/humble/rclpy

# USD
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/kit/exts/omni.usd:$ISAACSIM_PATH/kit/exts/omni.usd.core:$ISAACSIM_PATH/kit/exts/omni.usd.config:$ISAACSIM_PATH/kit/exts/omni.usd.libs:$ISAACSIM_PATH/kit/exts/omni.kit.usd.layers:$ISAACSIM_PATH/kit/exts/usdrt.scenegraph:$ISAACSIM_PATH/kit/exts/omni.usd.schema.semantics:$ISAACSIM_PATH/kit/exts/omni.usd.schema.audio:$ISAACSIM_PATH/extsPhysics/omni.usd.schema.physx


# ADD THIS WILL POTENTIALLY CAUSE CONFLICT WITH PKGS ALREADY INSTALLED IN CONDA ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAACSIM_PATH/.:$ISAACSIM_PATH/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAACSIM_PATH/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAACSIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAACSIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAACSIM_PATH/kit:$ISAACSIM_PATH/kit/kernel/plugins:$ISAACSIM_PATH/kit/libs/iray:$ISAACSIM_PATH/kit/plugins:$SCRIPT_DIR/kit/plugins/carb_gfx:$ISAACSIM_PATH/kit/plugins/bindings-python:$ISAACSIM_PATH/kit/plugins/rtx:$ISAACSIM_PATH/kit/plugins/gpu.foundation:/$ISAACSIM_PATH/kit/kernel/plugins

# USD libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAACSIM_PATH/kit/exts/omni.usd.libs/bin:$ISAACSIM_PATH/kit/exts/omni.usd.schema.audio/bin:$ISAACSIM_PATH/kit/exts/omni.usd.schema.semantics/bin:$ISAACSIM_PATH/kit/extscore/omni.client/bin/deps:$ISAACSIM_PATH/extsPhysics/omni.usd.schema.physx/bin:$ISAACSIM_PATH/kit/exts/omni.kit.renderer.imgui/bin

# CARB
# echo "$ISAACSIM_PATH/kit/libcarb.so" > /etc/ld.so.preload (NOT RECOMMENDED)
export LD_PRELOAD="$ISAACSIM_PATH/kit/libcarb.so $ISAACSIM_PATH/kit/kernel/plugins/libcarb.settings.plugin.so $ISAACSIM_PATH/kit/kernel/py/carb/_carb.cpython-310-x86_64-linux-gnu.so $ISAACSIM_PATH/kit/kernel/py/carb/settings/_settings.cpython-310-x86_64-linux-gnu.so"
