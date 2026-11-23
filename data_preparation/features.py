                                                                           

from html import unescape
from itertools import groupby, pairwise
from math import copysign, fsum, log1p, sqrt
import re
from statistics import mean, pstdev


EARLY_REVIEW_COUNT = 5
MILLISECONDS_PER_DAY = 86_400_000
STARS = (1, 2, 3, 4, 5)
_SURROGATE = re.compile(r"[\ud800-\udfff]")
_BREAK_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)

FEATURE_GROUPS = {
    "rating": (
        "rating_mean", "rating_sd", "rating_min", "rating_max",
        *(f"rating_{star}_count" for star in STARS),
        "early_low_star_share",
    ),
    "timing": (
        *(f"chronological_rating_{position}" for position in range(1, 6)),
        "rating_first_to_last_change",
        "rating_position_slope", "rating_position_fit_rmse",
        *(f"review_gap_{position}_days" for position in range(1, 5)),
        "early_window_days", "rating_day_slope", "rating_day_fit_rmse",
        "no_time_variation",
    ),
    "purchase": (
        "verified_purchase_count", "unverified_purchase_count",
        "missing_purchase_count", "verified_purchase_share",
    ),
    "text": (
        "title_missing_count", "body_missing_count", "both_text_missing_count",
        "all_early_text_missing",
        "title_chars_mean", "title_chars_max", "title_words_mean",
        "body_chars_mean", "body_chars_max", "body_words_mean",
    ),
}

SCALABLE_FIELDS = (
    *FEATURE_GROUPS["rating"],
    *(name for name in FEATURE_GROUPS["timing"] if name != "no_time_variation"),
    *FEATURE_GROUPS["purchase"],
    *(name for name in FEATURE_GROUPS["text"] if name != "all_early_text_missing"),
)

                                                                              
                                                                           
                                                                              
                                                                 
_TEXT_LENGTH_FIELDS = (
    "title_chars_mean", "title_chars_max", "title_words_mean",
    "body_chars_mean", "body_chars_max", "body_words_mean",
)
FEATURE_PRE_TRANSFORMS = {
    **{f"review_gap_{position}_days": "log1p" for position in range(1, EARLY_REVIEW_COUNT)},
    "early_window_days": "log1p",
    "rating_day_slope": "signed_log1p",
    **{name: "log1p" for name in _TEXT_LENGTH_FIELDS},
}

                                                                             
                                                                                
                                                                                
                                                                              
                                               
FEATURE_VIEWS = {
    "full": tuple(name for group in FEATURE_GROUPS.values() for name in group),
    "compact": (
        "rating_mean", "rating_sd", "early_low_star_share",
        "rating_first_to_last_change", "early_window_days", "rating_day_slope",
        "verified_purchase_count",
        "title_chars_mean", "body_chars_mean",
    ),
}

if not set(FEATURE_PRE_TRANSFORMS) <= set(SCALABLE_FIELDS):
    raise ValueError("Pre-transforms must name scalable fields only")
if not set(FEATURE_VIEWS["compact"]) <= set(FEATURE_VIEWS["full"]):
    raise ValueError("The compact view must be a subset of the contract columns")


def apply_pre_transform(name, value):
                                                                             

                                                                              
                                                                              
                               
       
    kind = FEATURE_PRE_TRANSFORMS.get(name)
    if kind is None or value is None:
        return value
    if kind == "log1p":
        if value < 0:
            raise ValueError(f"{name}: log1p requires a non-negative value, got {value}")
        return log1p(value)
    if kind == "signed_log1p":
        return copysign(log1p(abs(value)), value)
    raise ValueError(f"{name}: unknown pre-transform {kind!r}")


def normalize_review_text(value):
                                     

                                                                              
                                                                          
                                                                          
                                                                   
       
    if not isinstance(value, str):
        return ""
    plain = " ".join(_BREAK_TAG.sub(" ", unescape(value)).split())
    return _SURROGATE.sub("\ufffd", plain)


def _fit_line(x, ratings):
                                                            
    x_mean = mean(x)
    rating_mean = mean(ratings)
    x_centered = [value - x_mean for value in x]
    denominator = fsum(value * value for value in x_centered)
    slope = 0.0
    if denominator:
        slope = fsum(
            offset * (rating - rating_mean)
            for offset, rating in zip(x_centered, ratings)
        ) / denominator
    squared_residuals = fsum(
        (rating - rating_mean - slope * offset) ** 2
        for offset, rating in zip(x_centered, ratings)
    )
    return slope, sqrt(squared_residuals / len(ratings))


def make_features(early_reviews):
                                                                                

                                                                            
                                                                           
                                                                      
                                                                       
                                                                         
       
    reviews = list(early_reviews)
    if len(reviews) != EARLY_REVIEW_COUNT:
        raise ValueError("Feature extraction requires exactly five early reviews")

    for review in reviews:
        if type(review.get("timestamp")) is not int:
            raise ValueError("Early review timestamp must be integer milliseconds")
        if type(review.get("rating")) is not int or review["rating"] not in STARS:
            raise ValueError("Early review rating must be an integer from 1 to 5")

    reviews.sort(key=lambda review: review["timestamp"])
    ratings = [review["rating"] for review in reviews]
    counts = {star: ratings.count(star) for star in STARS}
    features = {
        "rating_mean": mean(ratings),
        "rating_sd": pstdev(ratings),
        "rating_min": min(ratings),
        "rating_max": max(ratings),
        **{f"rating_{star}_count": count for star, count in counts.items()},
        "early_low_star_share": (counts[1] + counts[2]) / EARLY_REVIEW_COUNT,
    }

    first_timestamp = reviews[0]["timestamp"]
    elapsed_days = [
        (review["timestamp"] - first_timestamp) / MILLISECONDS_PER_DAY
        for review in reviews
    ]
    chronological_ratings = []
    tied_positions = []
    text_pairs = []
    for _, group in groupby(reviews, key=lambda review: review["timestamp"]):
        tied_reviews = list(group)
        start = len(chronological_ratings)
        midpoint = start + (len(tied_reviews) - 1) / 2
        tied_positions.extend([midpoint] * len(tied_reviews))
        tied_mean = mean(review["rating"] for review in tied_reviews)
        chronological_ratings.extend([tied_mean] * len(tied_reviews))
        text_pairs.extend(sorted(
            (normalize_review_text(review.get("title")),
             normalize_review_text(review.get("text")))
            for review in tied_reviews
        ))

                                                                             
                                                                       
    position_slope, position_rmse = _fit_line(tied_positions, ratings)
    day_slope, day_rmse = _fit_line(elapsed_days, ratings)
    features.update({
        **{
            f"chronological_rating_{position}": value
            for position, value in enumerate(chronological_ratings, start=1)
        },
        "rating_first_to_last_change": chronological_ratings[-1] - chronological_ratings[0],
        "rating_position_slope": position_slope,
        "rating_position_fit_rmse": position_rmse,
        **{
            f"review_gap_{position}_days": right - left
            for position, (left, right) in enumerate(pairwise(elapsed_days), start=1)
        },
        "early_window_days": elapsed_days[-1],
        "rating_day_slope": day_slope,
        "rating_day_fit_rmse": day_rmse,
        "no_time_variation": int(elapsed_days[-1] == 0),
    })

    verified = sum(review.get("verified_purchase") is True for review in reviews)
    unverified = sum(review.get("verified_purchase") is False for review in reviews)
    known = verified + unverified
    features.update({
        "verified_purchase_count": verified,
        "unverified_purchase_count": unverified,
        "missing_purchase_count": EARLY_REVIEW_COUNT - known,
        "verified_purchase_share": verified / known if known else None,
    })

    titles, bodies = zip(*text_pairs)
    missing_both = sum(not title and not body for title, body in text_pairs)
    features.update({
        "title_missing_count": titles.count(""),
        "body_missing_count": bodies.count(""),
        "both_text_missing_count": missing_both,
        "all_early_text_missing": int(missing_both == EARLY_REVIEW_COUNT),
        "title_chars_mean": mean(map(len, titles)),
        "title_chars_max": max(map(len, titles)),
        "title_words_mean": mean(len(title.split()) for title in titles),
        "body_chars_mean": mean(map(len, bodies)),
        "body_chars_max": max(map(len, bodies)),
        "body_words_mean": mean(len(body.split()) for body in bodies),
        "text_document": "\n\n".join(
            "\n".join(value for value in pair if value)
            for pair in text_pairs
            if any(pair)
        ),
    })
    return features
