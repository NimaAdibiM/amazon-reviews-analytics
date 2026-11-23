import importlib.util
import sys
from pathlib import Path


def load_partner(name):
    path = Path(__file__).resolve().parents[1] / 'data_preparation' / f'{name}.py'
    qualified = f'feature_representation_partner_{name}'
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


features = load_partner('features')
split_scale = load_partner('split_scale')
