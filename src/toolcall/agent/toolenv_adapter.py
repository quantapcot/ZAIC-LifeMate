"""Organizer SDK loader; does not inspect simulator state or use private methods."""
from pathlib import Path
import sys


def open_train_env(root, sample_id):
    root=Path(root).resolve()
    sdk=root/'train'
    if str(sdk) not in sys.path: sys.path.insert(0,str(sdk))
    from zaic_tools import ToolEnv
    return ToolEnv('train',sample_id,root=root)
