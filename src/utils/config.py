import os
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._dict = config_dict
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
    
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def __getitem__(self, key: str) -> Any:
        val = self._dict[key]
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._dict:
            val = self._dict[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        return default

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def items(self):
        return self._dict.items()

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def load_config(config_path: str = 'config.yaml') -> Config:
    if not os.path.isabs(config_path):
        root = get_project_root()
        config_path = str(root / config_path)
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

def get_device(config: dict = None) -> torch.device:
    device_type = 'auto'
    if config and 'training' in config and 'device' in config['training']:
        device_type = config['training']['device']
        
    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_type == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
