"""Use the preparation modules that produced this project's tables."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREPARATION = ROOT / 'data_preparation' if (ROOT / 'data_preparation').is_dir() else ROOT


def load_partner(name):
    path = PREPARATION / f'{name}.py'
    if not path.is_file():
        raise FileNotFoundError(f'MISSING_PREPARATION_MODULE: {path}')
    qualified = f'{__package__}_partner_{name}'
    if qualified not in sys.modules:
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(qualified, None)
            raise
    return sys.modules[qualified]


features = load_partner('features')
split_scale = load_partner('split_scale')
                                                                              
full = tuple(name for block in features.FEATURE_GROUPS.values() for name in block)
views = getattr(features, 'FEATURE_VIEWS', {
    'full': full,
    'compact': ('rating_mean', 'rating_sd', 'early_low_star_share',
                'rating_first_to_last_change', 'early_window_days', 'rating_day_slope',
                'verified_purchase_count', 'title_chars_mean', 'body_chars_mean'),
})
if tuple(views['full']) != full or not set(views['compact']).issubset(full):
    raise ValueError('INCONSISTENT_PREPARATION_FEATURE_VIEWS')
