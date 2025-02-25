
import os
from typing_extensions import Dict, List, Optional, Type, Union, Callable, Tuple, TYPE_CHECKING
from enum import Enum

# Omniverse
import carb
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.articulations.articulation_gripper import ArticulationGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper
from isaacsim.robot.manipulators.grippers import ParallelGripper

if TYPE_CHECKING:
    from .omni_robot import OmniRobot

class Singleton(type):
    """
    Metaclass for singletons
    """

    _instances = {}
    """
    Dictionary of singleton child classes inheriting from this metaclass, keyed by child class objects.
    """
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# ENTITY MODEL NAMES
# ROBOTS
# MOBILES/VEHICLES
PR2_MODEL = "pr2"
TURTLEBOT_BURGER_MODEL = "turtlebot3_burger"
KAYA_MODEL = "kaya"

JETBOT_MODEL = "jetbot"

# QUADRUPLED
ANYMAL_C_MODEL = "anymal_c"
UNITREE_A1_MODEL = "a1"
UNITREE_GO1_MODEL = "go1"

# MANIPULATORS
DOFBOT_MODEL = "dofbot"
FRANKA_MODEL = "franka"
FRANKA_WITH_ALT_FINGERS_MODEL = "franka_alt_fingers"
UR5_MODEL = "ur5"
UR5E_MODEL = "ur5e"
UR10_MODEL = "ur10"
UR10_WITH_LONG_SUCTION_MODEL = "ur10_long_suction"
UR10_WITH_SHORT_SUCTION_MODEL = "ur10_short_suction"
UR10E_MODEL = "ur10e"
UR16E_MODEL = "ur16e"
COBOTTA_PRO_900_MODEL = "cobotta_pro_900"
FESTO_COBOT_MODEL = "festo_cobot"
KAWASAKI_RS080N_RG2_MODEL = "rs080n_onrobot_rg2"
JACO2_MODEL = "jaco2"
KINOVA_GEN3_MODEL = "kinova_gen3"
FANUC_CRX10IAL_MODEL = "Fanuc_crx10ial"
IIWA_MODEL = "iiwa"

# HANDS/GRIPPERS
ALLEGRO_HAND_MODEL = "allegro_hand"
LONG_SUCTION_GRIPPER_MODEL = "long_gripper"
SHORT_SUCTION_GRIPPER_MODEL = "short_gripper"
ROBOTIQ_2F85_MODEL = "robotiq_2f_85"
ROBOTIQ_2F140_MODEL = "robotiq_2f_140"
SHADOW_HAND_MODEL = "shadow_hand"

# HUMANOID
SIMPLE_HUMANOID_MODEL = "humanoid"

# LEGGED
EVOBOT_MODEL = "Evobot"

# COMPOUND ROBOTS
FRANKA_ROBOTIQ_2F_140_MODEL = f"{FRANKA_MODEL}_{ROBOTIQ_2F140_MODEL}"
UR5_ROBOTIQ_2F_85_MODEL = f"{UR5_MODEL}_{ROBOTIQ_2F85_MODEL}"
UR5E_ROBOTIQ_2F_140_MODEL = f"{UR5E_MODEL}_{ROBOTIQ_2F140_MODEL}"
UR10E_ROBOTIQ_2F_140_MODEL = f"{UR10E_MODEL}_{ROBOTIQ_2F140_MODEL}"
UR10E_ALLEGRO_MODEL = f"{UR10E_MODEL}_{ALLEGRO_HAND_MODEL}"
IIWA_ALLEGRO_MODEL = f"{IIWA_MODEL}_allegro"

# OBJECTS
SIMPLE_WAREHOUSE_MODEL = "warehouse"
FULL_WAREHOUSE_MODEL = "full_warehouse"
RUBIKS_CUBE_MODEL = "rubiks_cube"
SIMPLE_ROOM_MODEL = "simple_room"
IAI_KITCHEN_MODEL = "kitchen"
UR10_MOUNT_MODEL = "ur10_mount"

class IsaacTaskId(Enum):
    TARGET_FOLLOWING = 1
    PATH_PLANNING = 2
    SIMPLE_STACKING = 3
    MPC_CUROBO = 4
    PICK_PLACE_CUROBO = 5
    PICK_PLACE_PANDA = 6
    HANOI_TOWER = 7

class IsaacCommon(object, metaclass=Singleton):
    NUCLEUS_ASSETS_DIR: str = "Isaac/"
    """
    Prefix name of remote Nucleus assets directory
    """

    def __init__(self):
        current_child_dir: Callable[Tuple[str]] = lambda child_dir_list: (
            os.path.join(os.path.dirname(os.path.realpath(__file__)), *child_dir_list))

        #: Supported articulation models, only of which the articulation view & controller are loaded
        self.SUPPORTED_ARTICULATION_MODELS: List[str] = [UR5_MODEL, UR5E_MODEL, UR5_ROBOTIQ_2F_85_MODEL, UR5E_ROBOTIQ_2F_140_MODEL,
                                                         UR10_MODEL, UR10E_MODEL, UR10_WITH_LONG_SUCTION_MODEL, UR10_WITH_SHORT_SUCTION_MODEL, UR10E_ROBOTIQ_2F_140_MODEL,
                                                         UR10E_ALLEGRO_MODEL,
                                                         FRANKA_MODEL, FRANKA_WITH_ALT_FINGERS_MODEL,
                                                         IIWA_MODEL, IIWA_ALLEGRO_MODEL,
                                                         KINOVA_GEN3_MODEL, JACO2_MODEL, DOFBOT_MODEL,
                                                         PR2_MODEL]

        #: Supported robot models, only of which the kinematics models are loaded
        self.SUPPORTED_ROBOT_MODELS: List[str] = self.SUPPORTED_ARTICULATION_MODELS + [IAI_KITCHEN_MODEL]

        #: Modular robot models, constituted from separate body + gripper models
        self.MODULAR_ROBOT_MODELS: List[str] = [FRANKA_ROBOTIQ_2F_140_MODEL,
                                                UR5_ROBOTIQ_2F_85_MODEL, UR5E_ROBOTIQ_2F_140_MODEL, UR10E_ROBOTIQ_2F_140_MODEL,
                                                UR10E_ALLEGRO_MODEL,
                                                IIWA_ALLEGRO_MODEL]

        #: Robot models having built-in gripper
        self.BULTIN_GRIPPER_ROBOT_MODELS: List[str] = [PR2_MODEL,
                                                       FRANKA_MODEL, FRANKA_WITH_ALT_FINGERS_MODEL, FRANKA_ROBOTIQ_2F_140_MODEL,
                                                       KINOVA_GEN3_MODEL, JACO2_MODEL, DOFBOT_MODEL,
                                                       UR5_ROBOTIQ_2F_85_MODEL, UR5E_ROBOTIQ_2F_140_MODEL, UR10E_ROBOTIQ_2F_140_MODEL,
                                                       UR10E_ALLEGRO_MODEL, UR10_WITH_LONG_SUCTION_MODEL, UR10_WITH_SHORT_SUCTION_MODEL,
                                                       IIWA_ALLEGRO_MODEL]

        #: Path to the external directory where Isaac assets (urdf, cads, usd, etc.) are stored
        self.ISAAC_EXTERNAL_ASSETS_DIRECTORY: str = str(current_child_dir(("assets",)))

        #: Path to the external directory where Curobo assets (urdf, cads, usd, etc.) are stored
        self.CUROBO_EXTERNAL_ASSETS_DIRECTORY: str = self.ISAAC_EXTERNAL_ASSETS_DIRECTORY

        #: Path to the external directory where Curobo configs are stored
        self.CUROBO_EXTERNAL_CONFIGS_DIRECTORY: str = str(current_child_dir(("configs", "curobo")))

        #: Path to the external directory where RMP configs are stored
        self.RMP_EXTERNAL_CONFIGS_DIRECTORY: str = str(current_child_dir(("configs", "rmp")))

        #: Path to the external directory where path planning configs are stored
        self.PATH_PLAN_EXTERNAL_CONFIGS_DIRECTORY: str = str(current_child_dir(("configs", "path_plan")))

        #: Dictionary of USD paths of robot and object models, keyed by model names. Ref: https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_robots.html
        self.USD_PATHS: Dict[str, str] = {model_name: f"{IsaacCommon.NUCLEUS_ASSETS_DIR}/{usd_path}"
                                                      if "/" in usd_path else usd_path
                                                      for model_name, usd_path in
            {
                # Robots
                # Mobiles/Vehicles
                PR2_MODEL: "pr2.usd",
                TURTLEBOT_BURGER_MODEL: "Robots/Turtlebot/turtlebot3_burger.usd",
                KAYA_MODEL: "Robots/Kaya/kaya.usd",
                JETBOT_MODEL: "Robots/Jetbot/jetbot.usd",

                # Quadrupled
                ANYMAL_C_MODEL: "Robots/ANYbotics/anymal_c.usd",
                UNITREE_A1_MODEL: "Robots/Unitree/a1.usd",
                UNITREE_GO1_MODEL: "Robots/Unitree/go1.usd",

                # Manipulators
                DOFBOT_MODEL: "Robots/Dofbot/dofbot.usd",
                FRANKA_MODEL: "Robots/Franka/franka.usd",
                FRANKA_WITH_ALT_FINGERS_MODEL: "Robots/Franka/franka_alt_fingers.usd",
                UR5_MODEL: "Robots/UniversalRobots/ur5/ur5.usd",
                UR5E_MODEL: "Robots/UniversalRobots/ur5e/ur5e.usd",
                UR5_ROBOTIQ_2F_85_MODEL: "ur5_robotiq_2f_85.usd",
                UR5E_ROBOTIQ_2F_140_MODEL: "ur5e_robotiq_2f_140.usdc",
                UR10_MODEL: "Robots/UR10/ur10.usd",
                UR10_WITH_LONG_SUCTION_MODEL: "Robots/UR10/ur10_long_suction.usd",
                UR10_WITH_SHORT_SUCTION_MODEL: "Robots/UR10/ur10_short_suction.usd",
                UR10E_MODEL: "Robots/UniversalRobots/ur10e/ur10e.usd",
                UR10E_ROBOTIQ_2F_140_MODEL: "ur10e_robotiq_2f_140.usd",
                UR16E_MODEL: "Robots/UniversalRobots/ur16e/ur16e.usd",
                IIWA_MODEL: "iiwa.usd",
                COBOTTA_PRO_900_MODEL: "Robots/Denso/cobotta_pro_900.usd",
                FESTO_COBOT_MODEL: "Robots/Festo/FestoCobot/festo_cobot.usd",
                KAWASAKI_RS080N_RG2_MODEL: "Robots/Kawasaki/RS080N/rs080n_onrobot_rg2.usd",
                JACO2_MODEL: "Robots/Kinova/Jaco2/j2n7s300_instanceable.usd",
                KINOVA_GEN3_MODEL: "Robots/Kinova/Gen3/gen3n7_instanceable.usd",
                FANUC_CRX10IAL_MODEL: "Robots/Fanuc/CRX10IAL/crx10ial.usd",

                # Grippers
                ALLEGRO_HAND_MODEL: "Robots/AllegroHand/allegro_hand.usd",
                #IIWA_ALLEGRO_MODEL: "iiwa_allegro.usd",
                LONG_SUCTION_GRIPPER_MODEL: "Robots/UR10/Props/long_gripper.usd",
                SHORT_SUCTION_GRIPPER_MODEL: "Robots/UR10/Props/short_gripper.usd",
                ROBOTIQ_2F85_MODEL: "Robots/Robotiq/2F-85/2f85_instanceable.usd",
                ROBOTIQ_2F140_MODEL: "robotiq_2f_140.usdc", #"Robots/Robotiq/2F-140/2f140_instanceable.usd",
                SHADOW_HAND_MODEL: "Robots/ShadowHand/shadow_hand_instanceable.usd",

                # Humanoid
                SIMPLE_HUMANOID_MODEL: "Robots/Humanoid/humanoid_instanceable.usd",

                # Legged
                EVOBOT_MODEL: "Robots/Evobot/evobot.usd",

                # Objects
                UR10_MOUNT_MODEL: "Props/Mounts/ur10_mount.usd",
                SIMPLE_ROOM_MODEL: "Environments/Simple_Room/simple_room.usd",
                SIMPLE_WAREHOUSE_MODEL: "Environments/Simple_Warehouse/warehouse.usd",
                FULL_WAREHOUSE_MODEL: "Environments/Simple_Warehouse/full_warehouse.usd",
                RUBIKS_CUBE_MODEL: "Props/Rubiks_Cube/rubiks_cube.usd"
            }.items()
        }

        # Import specific robot classes here to avoid circular dependence
        from .pr2 import PR2
        from .panda import Panda
        from .ur10 import UR10
        from .ur5 import UR5
        from .ur5e_2f_140 import UR5E_2F_140
        from .dofbot import DofBot
        from .iiwa import IIWA
        from .iiwa_allegro import IIWA_ALLEGRO
        if TYPE_CHECKING:
            from .omni_robot import OmniRobot

        #: Dictionary of model-wise robot classes, keyed by robot model names
        self.ROBOT_CLASSES: Dict[str, Type[OmniRobot]] = {
            PR2_MODEL: PR2,
            FRANKA_MODEL: Panda,
            FRANKA_WITH_ALT_FINGERS_MODEL: Panda,
            UR5_MODEL: UR5,
            UR5E_MODEL: UR5,
            UR5E_ROBOTIQ_2F_140_MODEL: UR5E_2F_140,
            UR10_MODEL: UR10,
            UR10E_MODEL: UR10,
            UR10E_ROBOTIQ_2F_140_MODEL: UR10,
            UR10_WITH_LONG_SUCTION_MODEL: UR10,
            UR10_WITH_SHORT_SUCTION_MODEL: UR10,
            UR10E_ALLEGRO_MODEL: UR10,
            DOFBOT_MODEL: DofBot,
            IIWA_MODEL: IIWA,
            IIWA_ALLEGRO_MODEL: IIWA_ALLEGRO,
        }

        #: Dictionary of end-effector-mounting frame names of robots, keyed by robot models
        self.ROBOT_EE_MOUNT_FRAME_NAMES: Dict[str, str] = {
            UR5_MODEL: "tool0",
            UR5E_MODEL: "tool0",
            UR5_ROBOTIQ_2F_85_MODEL: "tool0",
            UR5E_ROBOTIQ_2F_140_MODEL: "tool0",
            UR10_MODEL: "tool0",
            UR10E_MODEL: "tool0",
            UR10E_ALLEGRO_MODEL: "tool0",
            UR10E_ROBOTIQ_2F_140_MODEL: "tool0",
            IIWA_MODEL: "iiwa7_link_ee",
        }

        #: Dictionary of model-wise gripper classes, keyed by gripper model names
        self.GRIPPER_CLASSES: Dict[str, Type[Union[Gripper, ArticulationGripper]]] = {
            SHORT_SUCTION_GRIPPER_MODEL: SurfaceGripper,
            LONG_SUCTION_GRIPPER_MODEL: SurfaceGripper,
            PR2_MODEL: ParallelGripper,
            FRANKA_MODEL: ParallelGripper,
            FRANKA_WITH_ALT_FINGERS_MODEL: ParallelGripper,
            FRANKA_ROBOTIQ_2F_140_MODEL: ParallelGripper,
            DOFBOT_MODEL: ParallelGripper,
            ROBOTIQ_2F85_MODEL: ParallelGripper,
            ROBOTIQ_2F140_MODEL: ParallelGripper,
            UR5_ROBOTIQ_2F_85_MODEL: ParallelGripper,
            UR5E_ROBOTIQ_2F_140_MODEL: ParallelGripper,
            UR10_WITH_LONG_SUCTION_MODEL: SurfaceGripper,
            UR10_WITH_SHORT_SUCTION_MODEL: SurfaceGripper,
            UR10E_ROBOTIQ_2F_140_MODEL: ParallelGripper,
            ALLEGRO_HAND_MODEL: ArticulationGripper,
            UR10E_ALLEGRO_MODEL: ArticulationGripper,
            IIWA_ALLEGRO_MODEL: ArticulationGripper,
            SHADOW_HAND_MODEL: ArticulationGripper
        }

        #: Dictionary of gripper-mounting frame names of grippers, keyed by gripper or compound robot-gripper models
        self.GRIPPER_MOUNT_FRAME_NAMES: Dict[str, str] = {
            ROBOTIQ_2F85_MODEL: "robotiq_arg2f_base_link",
            ROBOTIQ_2F140_MODEL: "robotiq_arg2f_base_link",
            UR5_ROBOTIQ_2F_85_MODEL: "robotiq_arg2f_base_link",
            UR5E_ROBOTIQ_2F_140_MODEL: "robotiq_arg2f_base_link",
            ALLEGRO_HAND_MODEL: "allegro_mount",
            IIWA_ALLEGRO_MODEL: "allegro_mount",
            UR10E_ALLEGRO_MODEL: "allegro_mount",
            SHADOW_HAND_MODEL: "robot0_hand_mount"
        }
        ROBOTIQ_HAND_JOINTS = ['finger_joint', 'right_outer_knuckle_joint',
                               'left_inner_finger_joint', 'right_inner_finger_joint',
                               'left_inner_finger_knuckle_joint', 'right_inner_finger_knuckle_joint']
        ALLEGRO_HAND_JOINTS = ["index_joint_0",
                               "index_joint_1",
                               "index_joint_2",
                               "index_joint_3",
                               "middle_joint_0",
                               "middle_joint_1",
                               "middle_joint_2",
                               "middle_joint_3",
                               "ring_joint_0",
                               "ring_joint_1",
                               "ring_joint_2",
                               "ring_joint_3",
                               "thumb_joint_0",
                               "thumb_joint_1",
                               "thumb_joint_2",
                               "thumb_joint_3"]

        #: Dictionary of model-wise gripper finger joint names, keyed by robot or gripper model names
        self.GRIPPER_FINGER_JOINTS: Dict[str, List[str]] = {
            #PR2_MODEL: ["r_gripper_l_finger_joint", "r_gripper_r_finger_joint"],
            PR2_MODEL: ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint',
                        'l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
                        'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_tip_joint',
                        'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_tip_joint'],
            FRANKA_MODEL: ["panda_finger_joint1", "panda_finger_joint2"],
            FRANKA_WITH_ALT_FINGERS_MODEL: ["panda_finger_joint1", "panda_finger_joint2"],
            ROBOTIQ_2F85_MODEL: ROBOTIQ_HAND_JOINTS,
            ROBOTIQ_2F140_MODEL: ROBOTIQ_HAND_JOINTS,
            UR5_ROBOTIQ_2F_85_MODEL: ROBOTIQ_HAND_JOINTS,
            UR5E_ROBOTIQ_2F_140_MODEL: ROBOTIQ_HAND_JOINTS,
            UR10E_ROBOTIQ_2F_140_MODEL: ROBOTIQ_HAND_JOINTS,
            ALLEGRO_HAND_MODEL: ALLEGRO_HAND_JOINTS,
            IIWA_ALLEGRO_MODEL: ALLEGRO_HAND_JOINTS,
            UR10E_ALLEGRO_MODEL: ALLEGRO_HAND_JOINTS,
            DOFBOT_MODEL: ["Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint"]
        }

        self.ROBOT_BASE_BODY_MODEL_NAMES: Dict[str, str] = {
            UR5_MODEL: UR5_MODEL,
            UR5_ROBOTIQ_2F_85_MODEL: UR5_MODEL,
            UR5E_MODEL: UR5E_MODEL,
            UR5E_ROBOTIQ_2F_140_MODEL: UR5E_MODEL,
            UR10_MODEL: UR10_MODEL,
            UR10_WITH_LONG_SUCTION_MODEL: UR10_MODEL,
            UR10_WITH_SHORT_SUCTION_MODEL: UR10_MODEL,
            UR10E_MODEL: UR10E_MODEL,
            UR10E_ALLEGRO_MODEL: UR10E_MODEL,
            UR10E_ROBOTIQ_2F_140_MODEL: UR10E_MODEL,
            FRANKA_MODEL: FRANKA_MODEL,
            FRANKA_ROBOTIQ_2F_140_MODEL: FRANKA_MODEL,
            IIWA_MODEL: IIWA_MODEL,
            IIWA_ALLEGRO_MODEL: IIWA_MODEL
        }

        self.ROBOT_GRIPPER_MODEL_NAMES: Dict[str, str] = {
            ROBOTIQ_2F85_MODEL: ROBOTIQ_2F85_MODEL,
            UR5_ROBOTIQ_2F_85_MODEL: ROBOTIQ_2F85_MODEL,
            ROBOTIQ_2F140_MODEL: ROBOTIQ_2F140_MODEL,
            UR5E_ROBOTIQ_2F_140_MODEL: ROBOTIQ_2F140_MODEL,
            UR10E_ROBOTIQ_2F_140_MODEL: ROBOTIQ_2F140_MODEL,
            FRANKA_ROBOTIQ_2F_140_MODEL: ROBOTIQ_2F140_MODEL,
            UR10_WITH_LONG_SUCTION_MODEL: LONG_SUCTION_GRIPPER_MODEL,
            UR10_WITH_SHORT_SUCTION_MODEL: SHORT_SUCTION_GRIPPER_MODEL,
            ALLEGRO_HAND_MODEL: ALLEGRO_HAND_MODEL,
            IIWA_ALLEGRO_MODEL: ALLEGRO_HAND_MODEL,
            UR10E_ALLEGRO_MODEL: ALLEGRO_HAND_MODEL,
            SHADOW_HAND_MODEL: SHADOW_HAND_MODEL
        }

        self.CUROBO_MODEL_CONFIGS_FILENAMES: Dict[str, str] = {
            PR2_MODEL: "pr2.yml",
            UR5_MODEL: "ur5e.yml",
            UR5E_MODEL: "ur5e.yml",
            UR5_ROBOTIQ_2F_85_MODEL: "ur5e_robotiq_2f_85.yml",
            UR5E_ROBOTIQ_2F_140_MODEL: "ur5e_robotiq_2f_140.yml",
            UR10_MODEL: "ur10.yml",
            UR10E_MODEL: "ur10e.yml",
            UR10E_ROBOTIQ_2F_140_MODEL: "ur10e.yml",
            UR10_WITH_LONG_SUCTION_MODEL: "ur10.yml",
            UR10_WITH_SHORT_SUCTION_MODEL: "ur10.yml",
            UR10E_ALLEGRO_MODEL: "ur10e_allegro_hand.yml",
            FRANKA_MODEL: "franka.yml",
            IIWA_MODEL: "iiwa.yml",
            IIWA_ALLEGRO_MODEL: "iiwa_allegro.yml",
            IAI_KITCHEN_MODEL: "iai_kitchen.yml",
            "kitchen": "kitchen.yml",
            "milk_box": "milk_box.yml",
            "jeroen_cup": "jeroen_cup.yml",
            "bowl": "bowl.yml",
            "pitcher": "pitcher.yml",
            "breakfast_cereal": "breakfast_cereal.yml",
            "Static_MilkPitcher": "Static_MilkPitcher.yml"
        }

    def get_nucleus_assets_root_path(self) -> str:
        assets_root_path = get_assets_root_path()
        if assets_root_path:
            return assets_root_path
        else:
            carb.log_error("Could not find Isaac Sim Nucleus assets folder")
            return ""

    def get_usd_full_path(self, usd_path: str) -> str:
        if not usd_path:
            return ""
        return os.path.join(self.get_nucleus_assets_root_path(), usd_path) if usd_path.startswith(IsaacCommon.NUCLEUS_ASSETS_DIR) \
                else os.path.join(self.ISAAC_EXTERNAL_ASSETS_DIRECTORY, usd_path)

    def has_usd(self, entity_model_name: str) -> bool:
        return entity_model_name in self.USD_PATHS

    def get_entity_default_full_usd_path(self, entity_model_name: str) -> str:
        return self.get_usd_full_path(self.USD_PATHS.get(entity_model_name))

    def is_supported_robot_model(self, entity_model_name: str) -> bool:
        """
        Check if the given robot model is supported as an robot in Omniverse
        :param entity_model_name: Name of an entity model. Eg: 'pr2'
        """
        return entity_model_name in self.SUPPORTED_ROBOT_MODELS

    def is_supported_articulation_model(self, entity_model_name: str) -> bool:
        """
        Check if the given entity model is supported as an articulation in Omniverse
        :param entity_model_name: Name of an entity model. Eg: 'pr2'
        """
        return entity_model_name in self.SUPPORTED_ARTICULATION_MODELS

    def get_robot_base_body_model_name(self, robot_model_name: str) -> str:
        return self.ROBOT_BASE_BODY_MODEL_NAMES.get(robot_model_name)

    def get_robot_class(self, robot_model_name: str) -> Type[Robot]:
        from .omni_robot import OmniRobot
        return self.ROBOT_CLASSES[robot_model_name] if robot_model_name in self.ROBOT_CLASSES else OmniRobot

    def is_modular_robot_models(self, robot_model_name: str) -> bool:
        return robot_model_name in self.MODULAR_ROBOT_MODELS

    def get_robot_gripper_model_name(self, robot_model_name: str) -> str:
        return self.ROBOT_GRIPPER_MODEL_NAMES.get(robot_model_name)

    def get_gripper_class(self, gripper_model_name: str) -> Optional[Type[Union[Gripper, ArticulationGripper]]]:
        return self.GRIPPER_CLASSES.get(gripper_model_name)

    def is_gripper_model_articulation(self, gripper_model_name: str) -> bool:
        gripper_class = self.get_gripper_class(gripper_model_name)
        return gripper_class and issubclass(gripper_class, ArticulationGripper)

    def has_builtin_gripper_model(self, robot_model_name: str) -> bool:
        return self.is_supported_robot_model(robot_model_name) and \
            robot_model_name in self.BULTIN_GRIPPER_ROBOT_MODELS

    def get_gripper_finger_joint_names(self, gripper_model_name: str) -> List[str]:
        return self.GRIPPER_FINGER_JOINTS.get(gripper_model_name)

    def get_gripper_mount_frame_name(self, robot_model_name: str) -> str:
        return self.GRIPPER_MOUNT_FRAME_NAMES.get(robot_model_name)

    def get_robot_ee_mount_frame_name(self, robot_model_name: str) -> str:
        return self.ROBOT_EE_MOUNT_FRAME_NAMES.get(robot_model_name)
