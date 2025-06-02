from typing import List, Optional
from dataclasses import dataclass, field, fields

# Hydra
from hydra.core.config_store import ConfigStore

# Cognarai
from cognarai.mppi.mppi_torch import MPPIConfig
from cognarai.mppi.mppi_env import MPPIEnvCfg, TASK_NAME_ALLEGRO_INHAND, TASK_NAME_FRANKA_CABINET

@dataclass
class MPPITaskBaseCfg:
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    render: bool = False
    n_steps: int = 100
    goal: Optional[List[float]] = None
    nx: int = 0
    entity_names: Optional[List[str]] = None
    initial_actor_positions: Optional[List[List[float]]] = None

@dataclass
class MPPITaskCfg(MPPITaskBaseCfg):
    env: MPPIEnvCfg = field(default_factory=MPPIEnvCfg)

    @classmethod
    def from_parent(cls, parent: MPPITaskBaseCfg, **kwargs):
        parent_field_names = {f.name for f in fields(MPPITaskBaseCfg)}
        parent_data = {name: getattr(parent, name) for name in parent_field_names}
        return cls(**parent_data, **kwargs)

    def __post_init__(self):
        self.nx = self.env.action_space

# NOTE: omegaconf, used by hydra, still does not support Callable, Literal, which are used in lots of IsaacLab cfg classes
# -> So, cannot preload a MPPIEnvCfg into memory here
cs = ConfigStore.instance()
cs.store(group="mppi", name=f"{TASK_NAME_ALLEGRO_INHAND}_mppi", node=MPPIConfig)
cs.store(name=TASK_NAME_ALLEGRO_INHAND, node=MPPITaskBaseCfg)
cs.store(group="mppi", name=f"{TASK_NAME_FRANKA_CABINET}_mppi", node=MPPIConfig)
cs.store(name=TASK_NAME_FRANKA_CABINET, node=MPPITaskBaseCfg)


from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
def load_config(name: str, config_path: str) -> DictConfig:
    with initialize(config_path=config_path):
        task_cfg = compose(config_name=name)
        #print(OmegaConf.to_yaml(task_cfg))
    return task_cfg

def load_task_config(task_name: str) -> MPPITaskBaseCfg:
    return load_config(task_name, config_path=f"../cfg/task")

def load_mppi_config(task_name: str) -> MPPIConfig:
    return load_config(f"{task_name}_mppi", config_path=f"../cfg/task/mppi")
