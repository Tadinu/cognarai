from typing import Optional

# USD
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, UsdShade, UsdLux
from omni.physx.scripts import physicsUtils


# Ref: https://forums.developer.nvidia.com/t/how-to-eliminate-the-rotation-in-spherical-joints/337224
class SphericalD6Joint:
    def __init__(self, stage, joint_path: Sdf.Path,
                 body0_path: Optional[Sdf.Path] = None, body1_path: Optional[Sdf.Path] = None,
                 local_pos0: Optional[Gf.Vec3f] = Gf.Vec3f(0, 0, 0),
                 local_pos1: Optional[Gf.Vec3f] = Gf.Vec3f(0, 0, 0),
                 damping=0.0, stiffness=0.0):
        """
        Create a D6 joint that behaves like a spherical joint (pure rotation only)

        Args:
            stage: USD stage
            joint_path: Path for the joint prim
            body0_path: Path to first rigid body
            body1_path: Path to second rigid body
            local_pos0: Local position on body0
            local_pos1: Local position on body1
            damping: Joint damping for rotational axes
            stiffness: Joint stiffness for rotational axes
        """
        self.stage = stage
        self.joint_path: Sdf.Path = joint_path
        self.body0_path = body0_path or '/World'
        self.body1_path = body1_path
        self.local_pos0 = local_pos0
        self.local_pos1 = local_pos1
        self.damping = damping
        self.stiffness = stiffness
        self.joint: UsdPhysics.Joint = None
        self._create_joint()

    def _create_joint(self):
        # https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html
        # Create a D6 joint
        self.joint = UsdPhysics.Joint.Define(self.stage, self.joint_path)
        joint_prim = self.joint.GetPrim()

        # Lock all translational dofs
        for axis in [UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ]:
            limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limitAPI.CreateLowAttr(0.0)
            limitAPI.CreateHighAttr(0.0)

        # Set the bodies to connect & local pose
        self.joint.GetBody0Rel().SetTargets([self.body0_path])
        self.joint.CreateLocalPos0Attr().Set(self.local_pos0)
        self.joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        if self.body1_path:
            self.joint.GetBody1Rel().SetTargets([self.body1_path])
            self.joint.CreateLocalPos1Attr().Set(self.local_pos1)
            self.joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        # Configure joint drives for rotational axes
        # These will allow free rotation around specific axes
        for axis in [UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ]:
            # Configure drive for damping/stiffness
            driveAPI = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            driveAPI.CreateTypeAttr(UsdPhysics.Tokens.force)
            driveAPI.CreateDampingAttr(self.damping)
            driveAPI.CreateStiffnessAttr(self.stiffness)

    def get_joint(self):
        return self.joint


def mfr_add_free_joint(prim_path: str, stage, base_body_name: str,
                       prim_pos: list[float],
                       damping: float = 200, stiffness: float = 0.0):
    # Create the spherical joint for cuboid
    prim_path = Sdf.Path(prim_path)
    base_body_path = f"{prim_path}/{base_body_name}"
    joint_path = prim_path.AppendChild("spherical_joint")
    SphericalD6Joint(stage, joint_path,
                     body0_path=prim_path,
                     body1_path=base_body_path,
                     damping=damping,  # Add some damping to prevent excessive oscillation
                     stiffness=stiffness)  # No stiffness - free rotation

    # Create a fixed joint to anchor the prim to the world
    anchor_joint_path = prim_path.AppendChild("fixed_joint")
    anchor_joint = UsdPhysics.FixedJoint.Define(stage, anchor_joint_path)

    # Set the cuboid as body1 (body0 will be the world/ground)
    anchor_joint.GetBody0Rel().SetTargets(['/World'])
    anchor_joint.GetBody1Rel().SetTargets([base_body_path])

    # Set the local position for the anchor joint
    anchor_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(prim_pos))
    anchor_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
