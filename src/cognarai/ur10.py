import os
from typing import Optional, Sequence, Union

import numpy as np

# Omniverse
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf, PhysxSchema

# Isaac-Interface
from .omni_robot import OmniRobot
from .isaac_common import *


class UR10(OmniRobot):
    """[summary]

    Args:
        prim_path (str): [description]
        articulation_root_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        description_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        robot_model_name: str,
        description_path: str,
        prim_path: str,
        name: str,
        articulation_root_path: str,
        end_effector_prim_name: Optional[str] = None,
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        attach_extra_gripper: bool = True,
        gripper_model_name: Optional[str] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        gripper_fingers_deltas: Optional[np.ndarray] = None,
        visible: bool = True
    ) -> None:
        if gripper_model_name is None:
            gripper_model_name = SHORT_SUCTION_GRIPPER_MODEL
        super().__init__(robot_model_name=robot_model_name,
                         description_path=description_path,
                         prim_path=prim_path, name=name, articulation_root_path=articulation_root_path,
                         end_effector_prim_name=end_effector_prim_name,
                         position=position,
                         translation=translation,
                         orientation=orientation,
                         scale=scale,
                         attach_extra_gripper=attach_extra_gripper,
                         gripper_model_name=gripper_model_name,
                         gripper_open_position=gripper_open_position,
                         gripper_closed_position=gripper_closed_position,
                         gripper_fingers_deltas=gripper_fingers_deltas,
                         rmp_policy_name="RMPflow",
                         rmp_policy_with_gripper_name="RMPflowSuction" if gripper_model_name == SHORT_SUCTION_GRIPPER_MODEL
                                                      else "RMPflowCortex",
                         visible=visible)
        if robot_model_name.startswith(UR10E_MODEL):
            self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                        "universal_robots", "ur10e", "rmpflow",
                                                        "ur10e_robot_description.yaml")
            import pathlib
            assert pathlib.Path(self.cspace_description_path).exists()
            self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                      "universal_robots", "ur10e", "ur10e.urdf")
            import pathlib
            assert pathlib.Path(self.lula_description_path).exists()
        else:
            if gripper_model_name == SHORT_SUCTION_GRIPPER_MODEL:
                self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                            "ur10", "rmpflow_suction", "ur10_robot_description.yaml")
                self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                          "ur10", "ur10_robot_suction.urdf")
            else:
                self.cspace_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                            "ur10", "rmpflow",
                                                            "ur10_robot_description.yaml")
                self.lula_description_path = os.path.join(self.isaac_common.RMP_EXTERNAL_CONFIGS_DIRECTORY,
                                                          "ur10", "ur10_robot.urdf")


    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        #self.set_joints_default_state(
        #    positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        #)

        stage = self.isaac.omni_stage
        ee_prim = stage.GetPrimAtPath(self.end_effector_prim_path)
        if self._gripper:
            # rorate about y by -90
            #ee_prim.set_local_pose(translation=np.ravel([0, 0, -3.0]), orientation=np.array([0.70711, 0.0, -0.70711, 0.0]))
            #ee_prim.AddTranslateOp().Set(value=(0, 0, 0))
            #self._gripper.prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3, False).Set(Gf.Vec3d(1.1834, 0.25614, 0.0116))
            #self._gripper.prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3, False).Set(Gf.Vec3d(0, 0, 0))
            #self._gripper.prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False).Set(Gf.Vec3d(1.0, 1.0, 1.0))
            #self._gripper.prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.String, False).Set(
            #    ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
            #)
            #self._gripper.prim.AddRotateXYZOp().Set(value=(0, 90, 0))

            #self._gripper.prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Double4, False).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            #self._gripper.prim.AddOrientOp.Set(Gf.Quatf(0.70711, 0.0, -0.70711, 0.0))
            #from omni.isaac.core.utils.prims import get_prim_parent
            #self.create_joint("Fixed", get_prim_parent(ee_prim), ee_prim)

            from omni.physx import get_physx_interface
            curr_transform = get_physx_interface().get_rigidbody_transformation(self.end_effector_prim_path)
            rv_success = curr_transform["ret_val"]
            if rv_success:
                curr_pos_f3 = curr_transform["position"]
                curr_pos = Gf.Vec3d(curr_pos_f3[0], curr_pos_f3[1], curr_pos_f3[2])
                curr_rot_f4 = curr_transform["rotation"]
                curr_rot_quat = Gf.Quatd(curr_rot_f4[3], curr_rot_f4[0], curr_rot_f4[1], curr_rot_f4[2])
