"""Phase-1-only, leakage-safe temporal split DESIGN, not a fitted/evaluated split.

All times are exact integer epochs in the SAME selected input unit. The
caller must supply scientifically chosen cutoffs; none are inferred from Y.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Cutoffs:
    train_end: int
    validation_end: int
    test_end: int

    def __post_init__(self):
        for field in ('train_end', 'validation_end', 'test_end'):
            if type(getattr(self, field)) is not int:
                raise TypeError(f'{field} must be an exact integer timestamp')
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError('Cutoffs must be strictly increasing')


def assign_matured_product(t5: int, t15: int | None, cutoffs: Cutoffs) -> str:
    """Assign a globally unique product by t5 AND its observed label maturity.

    Products whose outcome is not mature by the END of the corresponding
    time block are censored for that block, never silently sent to a later
    block with an earlier t5 or assigned Y=0. No access to text or labels.
    """
    if type(t5) is not int or (t15 is not None and type(t15) is not int):
        raise TypeError('t5/t15 must be same-unit integer epochs or missing t15')
    if t15 is not None and t15 < t5:
        raise ValueError('Outcome maturity before prediction time')
    if t5 <= cutoffs.train_end:
        block, end = 'train', cutoffs.train_end
    elif t5 <= cutoffs.validation_end:
        block, end = 'validation', cutoffs.validation_end
    elif t5 <= cutoffs.test_end:
        block, end = 'test', cutoffs.test_end
    else:
        return 'outside_observation_window'
    return block if t15 is not None and t15 <= end else f'{block}_label_not_mature'


def validate_unique_global_products(assignments):
    """Accept (global_parent, split) pairs; fail on any reused parent identifier."""
    seen = set()
    for parent, split in assignments:
        if not isinstance(parent, str) or not parent or parent != parent.strip():
            raise ValueError('Invalid product ID')
        if parent in seen:
            raise ValueError('Global product in multiple samples or splits')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Only mature outcomes may enter a model split')
        seen.add(parent)
    return len(seen)


def document():
    return dict(
        status='phase1_design_only_no_split_executed',
        unit='one globally unique raw parent_asin with one primary category',
        time_unit='same exact integer Unix epoch unit as input; cutoffs chosen and frozen before model fitting',
        cutoffs=dict(train_end=None, validation_end=None, test_end=None),
        mandatory_rules=[
            'Reconcile and exact-deduplicate globally BEFORE splitting; no product appears in two splits.',
            'Define t5 from 5th valid observed review and t15 from 15th, after boundary-tie exclusions.',
            'Time-block membership is determined ONLY by t5; a label is usable ONLY if t15 is no later than the end of its own block.',
            'Unmatured labels are censored, never treated as negative or transferred to a later block.',
            'Freeze block cutoffs and outcome observation window before model tuning; audit time-dependent selection.',
            'Preprocessing, imputation, vocabulary, IDF, category priors, text embeddings learned from corpus, scaling and thresholds fit on training only.',
            'Tune on validation; evaluate test only once after choices are frozen.',
            'Do not use future reviews 6–15, t15, crawl-time average_rating/rating_number, historical-unknown price or snapshot helpful_vote as predictors.',
            'Keep identical evaluation products/splits/model families and comparable tuning budgets for S versus ST.',
            'Use product-level paired loss and uncertainty with clustering/dependence considered; report each category and equal-category primary aggregate.',
        ],
        pending=['Choose calendar dates using actual follow-up coverage without looking at test outcome associations',
                 'Run split only with four full verified raw inputs and report dropout/censoring per category',
                 'Evaluate subgroup and category distribution shift, no fitted models in Phase 1'],
    )
