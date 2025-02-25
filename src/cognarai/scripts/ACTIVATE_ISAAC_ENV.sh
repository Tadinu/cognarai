workon isaac
source ./SOURCE_ALL.sh

#pushd $OMNI_KIT
#source setup_python_env.sh
#popd

pushd $ISAACSIM_PATH
source setup_python_env.sh
popd

$ISAACSIM_PYTHON -c "import os, torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(os.environ.get('CUDA_PATH')); print(torch.cuda.nccl.version()); \
					 print(torch._C._GLIBCXX_USE_CXX11_ABI)"

# CUROBO
#export PYTHONPATH=$PYTHONPATH:/home/tad/2_ISAAC/curobo/src

ISAACSIM_EXTS=$ISAACSIM_PATH/exts
ISAACSIM_EXTS_CACHE=$ISAACSIM_PATH/extscache
ISAACSIM_EXTS_PHYSICS=$ISAACSIM_PATH/extsPhysics
ISAACSIM_EXTS_DEPRECATED=$ISAACSIM_PATH/extsDeprecated
ISAACSIM_KIT_EXTS=$ISAACSIM_PATH/kit/exts
ISAACSIM_KIT_EXTS_CORE=$ISAACSIM_PATH/kit/extscore

# REF: [$ISAACSIM_PATH/setup_conda_env.sh -> setup_python_env.sh]
# DO NOT ADD [$ISAACSIM_PATH/kit/python/lib/python3.10/site-packages], which just confuse notebook server
# ISAACSIM
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/python_packages

# OMNI DEPRECATED
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS/omni.isaac.core_archive/pip_prebundle:$ISAACSIM_EXTS/omni.isaac.ml_archive/pip_prebundle:$ISAACSIM_EXTS/omni.pip.compute/pip_prebundle:$ISAACSIM_EXTS/omni.pip.cloud/pip_prebundle

# CORE EXTS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS/isaacsim.core.api:$ISAACSIM_EXTS/isaacsim.core.prims:$ISAACSIM_EXTS/isaacsim.core.simulation_manager:$ISAACSIM_EXTS/isaacsim.core.throttling:$ISAACSIM_EXTS/isaacsim.core.utils:$ISAACSIM_EXTS/isaacsim.core.version:$ISAACSIM_EXTS/isaacsim.core.nodes:$ISAACSIM_EXTS/isaacsim.core.includes

# EXTS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS/isaacsim.simulation_app:$ISAACSIM_EXTS/isaacsim.asset.gen.omap:$ISAACSIM_EXTS/isaacsim.robot.manipulators:$ISAACSIM_EXTS/isaacsim.robot_motion.lula:$ISAACSIM_EXTS/isaacsim.robot_motion.lula/pip_prebundle:$ISAACSIM_EXTS/isaacsim.robot_motion.motion_generation:$ISAACSIM_EXTS/isaacsim.robot_setup.assembler:$ISAACSIM_EXTS/isaacsim.robot.wheeled_robots:$ISAACSIM_EXTS/isaacsim.sensors.camera:$ISAACSIM_EXTS/isaacsim.sensors.physics:$ISAACSIM_EXTS/isaacsim.sensors.physx:$ISAACSIM_EXTS/isaacsim.sensors.rtx:$ISAACSIM_EXTS/isaacsim.storage.native:$ISAACSIM_EXTS/isaacsim.util.camera_inspector:$ISAACSIM_EXTS/isaacsim.util.clash_detection:$ISAACSIM_EXTS/isaacsim.util.debug_draw:$ISAACSIM_EXTS/isaacsim.util.merge_mesh:$ISAACSIM_EXTS/isaacsim.util.physics

# EXTSCACHE
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS_CACHE/omni.replicator.core-1.11.14+106.0.1.lx64.r.cp310:$ISAACSIM_EXTS_CACHE/omni.graph.scriptnode-1.19.1+106.0.0:$ISAACSIM_EXTS_CACHE/omni.kit.window.material_graph-1.8.15:$ISAACSIM_EXTS_CACHE/omni.kit.usd.layers-2.1.36+10a4b5c0.lx64.r.cp310:$ISAACSIM_EXTS_CACHE/omni.kit.commands-1.4.9+10a4b5c0.lx64.r.cp310:$ISAACSIM_EXTS_CACHE/omni.kit.actions.core-1.0.0+10a4b5c0.lx64.r.cp310

# EXTS PHYSICS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS_PHYSICS/omni.physics.tensors:$ISAACSIM_EXTS_PHYSICS/omni.physx

# EXTS DEPRECATED
export PYTHONPATH=$ISAACSIM_EXTS_DEPRECATED/omni.isaac.kit:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.core:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.gym:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.lula/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.exporter.urdf/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.kit.pip_archive/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.core_archive/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.ml_archive/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.pip.compute/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.pip.cloud/pip_prebundle:$PYTHONPATH:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.robot_assembler:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.occupancy_map:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.dynamic_control:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.nucleus:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.extension_templates:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.lula:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.manipulators:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.franka:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.universal_robots:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.surface_gripper:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.motion_generation:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.scene_blox:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.examples:$ISAACSIM_EXTS/omni.isaac.sensor:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.ui:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.version:$ISAACSIM_EXTS/omni.isaac.quadruped:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.core_nodes

# KIT KERNEL PY
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/kit/kernel/py:$PYTHONPATH:$ISAACSIM_PATH/kit/kernel/py/carb

# KIT
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_PATH/kit/plugins/bindings-python:$ISAACSIM_EXTS_CACHE/omni.kit.pip_archive-0.0.0+d02c707b.lx64.cp310/pip_prebundle

# KIT-EXTCORE
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_KIT_EXTS_CORE:$ISAACSIM_KIT_EXTS_CORE/omni.client:$ISAACSIM_KIT_EXTS_CORE/omni.assets.plugins:$ISAACSIM_KIT_EXTS_CORE/omni.kit.async_engine:$ISAACSIM_KIT_EXTS_CORE/omni.kit.registry.nucleus

# KIT-EXTS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_KIT_EXTS/omni.appwindow:$ISAACSIM_KIT_EXTS/omni.kit.commands:$ISAACSIM_KIT_EXTS/omni.kit.actions.core:$ISAACSIM_KIT_EXTS/omni.kit.actions.window:$ISAACSIM_KIT_EXTS/omni.timeline:$ISAACSIM_KIT_EXTS/omni.kit.numpy.common:$ISAACSIM_KIT_EXTS/omni.kit.material.library:$ISAACSIM_KIT_EXTS/omni.ui:$ISAACSIM_KIT_EXTS/omni.kit.renderer.imgui:$ISAACSIM_KIT_EXTS/omni.gpu_foundation:$ISAACSIM_KIT_EXTS/omni.kit.renderer.core:$ISAACSIM_KIT_EXTS/omni.ui.scene:$ISAACSIM_KIT_EXTS/omni.kit.helper.file_utils:$ISAACSIM_KIT_EXTS/omni.kit.widget.nucleus_connector:$ISAACSIM_KIT_EXTS/omni.kit.search_core:$ISAACSIM_KIT_EXTS/omni.kit.test:$ISAACSIM_KIT_EXTS/omni.usd.schema.audio:$ISAACSIM_KIT_EXTS/omni.usd.schema.audio:$ISAACSIM_KIT_EXTS/omni.kit.clipboard:$ISAACSIM_KIT_EXTS/omni.kit.widget.context_menu:$ISAACSIM_KIT_EXTS/omni.usd.schema.anim:$ISAACSIM_KIT_EXTS/omni.graph:$ISAACSIM_KIT_EXTS/omni.graph.tools:$ISAACSIM_KIT_EXTS/omni.usd.schema.omnigraph:$ISAACSIM_KIT_EXTS/omni.hydra.engine.stats:$ISAACSIM_KIT_EXTS/omni.syntheticdata

# URDF/MJCF IMPORTER/EXPORTER
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS_CACHE/isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310:$ISAACSIM_EXTS_CACHE/isaacsim.asset.importer.mjcf-2.3.3+106.3.0.lx64.r.cp310:$ISAACSIM_EXTS/isaacsim.asset.exporter.urdf/pip_prebundle

# OMNI.KIT.WINDOW-WIDGET
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_KIT_EXTS/omni.kit.window.file_importer:$ISAACSIM_KIT_EXTS/omni.kit.window.filepicker:$ISAACSIM_KIT_EXTS/omni.kit.widget.filebrowser:$ISAACSIM_KIT_EXTS/omni.kit.window.popup_dialog:$ISAACSIM_KIT_EXTS/omni.kit.widget.search_delegate:$ISAACSIM_KIT_EXTS/omni.kit.widget.nucleus_info:$ISAACSIM_KIT_EXTS/omni.kit.widget.versioning:$ISAACSIM_KIT_EXTS/omni.kit.widget.browser_bar:$ISAACSIM_KIT_EXTS/omni.kit.widget.path_field:$ISAACSIM_KIT_EXTS/omni.kit.notification_manager:$ISAACSIM_KIT_EXTS/omni.kit.menu.common:$ISAACSIM_KIT_EXTS/omni.kit.menu.utils:$ISAACSIM_KIT_EXTS/omni.kit.window.extensions:$ISAACSIM_KIT_EXTS/omni.kit.widget.graph:$ISAACSIM_KIT_EXTS/omni.kit.widget.filter:$ISAACSIM_KIT_EXTS/omni.kit.widget.options_menu:$ISAACSIM_KIT_EXTS/omni.kit.widget.options_button:$ISAACSIM_KIT_EXTS/omni.kit.window.property:$ISAACSIM_KIT_EXTS/omni.kit.widget.searchfield:$ISAACSIM_KIT_EXTS/omni.kit.widget.highlight_label:$ISAACSIM_KIT_EXTS/omni.kit.property.usd:$ISAACSIM_KIT_EXTS/omni.kit.widget.stage:$ISAACSIM_KIT_EXTS/omni.activity.core:$ISAACSIM_KIT_EXTS/omni.activity.ui:$ISAACSIM_KIT_EXTS/omni.kit.hotkeys.core:$ISAACSIM_KIT_EXTS/omni.kit.hotkeys.window:$ISAACSIM_KIT_EXTS/omni.kit.window.file_exporter

# TORCH
# Only if installing a custom torch, eg: 2.0.1
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS_CACHE/omni.pip.torch-2_0_1-2.0.2+105.1.lx64/torch-2-0-1

# ROS
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_EXTS/omni.isaac.ros_bridge:$ISAACSIM_EXTS/omni.isaac.ros_bridge/noetic/local/lib/python3.10/dist-packages:$ISAACSIM_EXTS/omni.isaac.ros2_bridge/humble/rclpy:$ISAACSIM_EXTS/omni.isaac.ros_bridge/noetic/local/lib/python3.10/dist-packages

# USD
# [$ISAACSIM_EXTS/omni.isaac.core/omni/isaac/core/utils/stage.py]: need to add [from usdrt import Usd]
export PYTHONPATH=$PYTHONPATH:$ISAACSIM_KIT_EXTS/omni.usd:$ISAACSIM_KIT_EXTS/omni.usd.core::$ISAACSIM_KIT_EXTS/omni.kit.usd_undo:$ISAACSIM_KIT_EXTS/omni.usd.config:$ISAACSIM_KIT_EXTS/omni.usd.libs:$ISAACSIM_KIT_EXTS/omni.kit.usd.layers:$ISAACSIM_KIT_EXTS/usdrt.scenegraph:$ISAACSIM_KIT_EXTS/omni.usd.schema.semantics:$ISAACSIM_KIT_EXTS/omni.usd.schema.audio:$ISAACSIM_EXTS_PHYSICS/omni.usd.schema.physx


# USD libs 1 - ADD THIS WILL POTENTIALLY CAUSE CONFLICT WITH PKGS ALREADY INSTALLED IN CONDA ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAACSIM_PATH/.:$ISAACSIM_EXTS_DEPRECATED/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAACSIM_EXTS_DEPRECATED/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAACSIM_EXTS_DEPRECATED/omni.isaac.lula/pip_prebundle:$ISAACSIM_EXTS_DEPRECATED/omni.exporter.urdf/pip_prebundle:$ISAACSIM_PATH/kit:$ISAACSIM_PATH/kit/kernel/plugins:$ISAACSIM_PATH/kit/libs/iray:$ISAACSIM_PATH/kit/plugins:$SCRIPT_DIR/kit/plugins/carb_gfx:$ISAACSIM_PATH/kit/plugins/bindings-python:$ISAACSIM_PATH/kit/plugins/rtx:$ISAACSIM_PATH/kit/plugins/gpu.foundation:/$ISAACSIM_PATH/kit/kernel/plugins

# USD libs 2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OMNI_KIT/exts/omni.usd.core-1.4.2+9321c814.lx64.r/bin:$ISAACSIM_EXTS_CACHE/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin:$ISAACSIM_EXTS_CACHE/omni.usd.schema.audio-0.0.0+d02c707b.lx64.r.cp310/bin:$ISAACSIM_EXTS_PHYSICS/omni.usd.schema.physx/bin

# CARB
# echo "$ISAACSIM_PATH/kit/libcarb.so" > /etc/ld.so.preload (NOT RECOMMENDED)
# carb.py: Need to add [from carb import settings]
#export LD_PRELOAD="$ISAACSIM_PATH/kit/libcarb.so $ISAACSIM_PATH/kit/kernel/plugins/libcarb.settings.plugin.so $ISAACSIM_PATH/kit/kernel/py/carb/_carb.cpython-310-x86_64-linux-gnu.so $ISAACSIM_PATH/kit/kernel/py/carb/settings/_settings.cpython-310-x86_64-linux-gnu.so"
export LD_PRELOAD="$ISAACSIM_PATH/kit/libcarb.so"
