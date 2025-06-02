from __future__ import annotations
from typing import List, TYPE_CHECKING
import yaml
from yaml import SafeLoader
from pathlib import Path
import os

import cognarai
if TYPE_CHECKING:
    from cognarai.mppi.mppi_env import EntityCfg

FILE_PATH = Path(__file__).parent.resolve()

def load_entity_cfgs(entities: List[str]) -> List[EntityCfg]:
    entity_cfgs = []
    for entity_name in entities:
        with open(
            f"{os.path.dirname(cognarai.__file__)}/mppi/cfg/entities/{entity_name}.yaml"
        ) as f:
            entity_cfgs.append(EntityCfg(**yaml.load(f, Loader=SafeLoader)))
    return entity_cfgs