"""Associations with explicit conditioning and no independence assumptions."""
from collections import Counter, defaultdict
import math
import random

from .stats import Distribution, frequency, rate
from .univariate import TEXT_STATES


class LogPairCorrelation:
    """Streaming Pearson on log1p(nonnegative lengths, votes), exact pair eligibility.

    This is an exploratory association, not a causal estimate. Validity is
    determined by the caller; an unpaired observation is never imputed.
    """
    def __init__(self):
        self.n = 0
        self.mean_x = self.mean_y = self.ss_x = self.ss_y = self.cross = 0.0

    def add(self, text_length, votes):
        if text_length is None or votes is None:
            return
        if text_length < 0 or votes < 0:
            raise ValueError("E39 paired values must be nonnegative")
        # math.log(int) handles integers too large to convert to finite float.
        x = math.log(text_length + 1) if text_length > 10**300 else math.log1p(text_length)
        y = math.log(votes + 1) if votes > 10**300 else math.log1p(votes)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("E39 invalid transformed input")
        self.n += 1
        dx = x - self.mean_x
        self.mean_x += dx / self.n
        dy = y - self.mean_y
        self.mean_y += dy / self.n
        self.ss_x += dx * (x - self.mean_x)
        self.ss_y += dy * (y - self.mean_y)
        self.cross += dx * (y - self.mean_y)
        return x, y

    def result(self, denominator):
        result = dict(valid_pairs=frequency(self.n, denominator),
                      transformation='natural log(1+x); only nonblank review body and valid helpful_vote',
                      measure='Pearson correlation on paired log1p values',
                      interpretation='observational, not causal; product/user clustering and exposure time unadjusted')
        if self.n < 2:
            result.update(correlation=None, reason='fewer_than_two_pairs')
        elif self.ss_x <= 0 or self.ss_y <= 0:
            result.update(correlation=None, reason='one_or_both_constant')
        else:
            corr = self.cross / math.sqrt(self.ss_x * self.ss_y)
            result.update(correlation=max(-1., min(1.,corr)) if math.isfinite(corr) else None,
                          reason='computed' if math.isfinite(corr) else 'numeric_failure')
        return result


class BoundedPairSample:
    """Reproducible, uniform reservoir of valid pairs; never exports raw IDs/text."""
    def __init__(self, cap=512, seed=1729):
        self.cap, self.seen, self.data = cap, 0, []
        self.random = random.Random(seed)

    def add(self, x, y, star):
        self.seen += 1
        point = dict(body_log1p=x, votes_log1p=y, star=star)
        if len(self.data) < self.cap:
            self.data.append(point)
        else:
            j = self.random.randrange(self.seen)
            if j < self.cap:
                self.data[j] = point

    def result(self):
        return dict(policy='uniform reservoir over all eligible paired raw review records',
                    seed=1729, cap=self.cap, eligible_pairs=self.seen,
                    sample_size=len(self.data), points=self.data,
                    note='Plots are sampled; Pearson uses ALL eligible pairs. No raw identifiers/text exported.')


class ReviewRelations:
    def __init__(self):
        self.stars = Counter()
        self.body_state = defaultdict(Counter)
        self.title_state = defaultdict(Counter)
        self.lengths = [Distribution() for _ in range(5)]
        self.votes = [Distribution() for _ in range(5)]
        self.verified = defaultdict(Counter)
        self.year_stars = defaultdict(Counter)
        self.year_all = Counter()
        self.year_body = Counter()
        self.year_title = Counter()
        self.year_verified = Counter()
        self.year_review_image = Counter()
        self.year_image_checked = Counter()
        self.image_by_star = Counter()
        self.star_image_checked = Counter()
        self.paired = LogPairCorrelation()
        self.paired_by_star = [LogPairCorrelation() for _ in range(5)]
        self.pair_sample = BoundedPairSample()

    def add(self, star, title_state, body_state, body_length, votes, verified, year, image_available=None):
        # Time-conditional availability does not discard invalid ratings.
        if year is not None:
            self.year_all[year] += 1
            self.year_body[year] += body_state == 'nonblank'
            self.year_title[year] += title_state == 'nonblank'
            self.year_verified[year] += verified is not None
            if image_available is not None:
                self.year_image_checked[year] += 1
                self.year_review_image[year] += bool(image_available)
        point = self.paired.add(body_length, votes)
        if point is not None:
            self.pair_sample.add(point[0], point[1], star)
        if star is None:
            return
        self.paired_by_star[star-1].add(body_length, votes)
        self.stars[star] += 1
        if image_available is not None:
            self.star_image_checked[star] += 1
            self.image_by_star[star] += bool(image_available)
        self.title_state[star][title_state] += 1
        self.body_state[star][body_state] += 1
        self.lengths[star-1].add(body_length)
        self.votes[star-1].add(votes)
        self.verified[star]["unusable" if verified is None else str(verified).lower()] += 1
        if year is not None:
            self.year_stars[year][star] += 1

    def result(self, raw_total):
        by_star = []
        for star in range(1, 6):
            n = self.stars[star]
            by_star.append(dict(star=star, valid_rating_records=n,
                                title=[dict(state=k, **frequency(self.title_state[star][k], n)) for k in TEXT_STATES],
                                body=[dict(state=k, **frequency(self.body_state[star][k], n)) for k in TEXT_STATES],
                                body_length=self.lengths[star-1].summary(n),
                                helpful_vote=self.votes[star-1].summary(n),
                                image_url_checked=frequency(self.star_image_checked[star],n),
                                review_image_url=frequency(self.image_by_star[star],self.star_image_checked[star]),
                                verified=[dict(state=k, **frequency(self.verified[star][k], n))
                                          for k in ("false", "true", "unusable")]))
        yearly = []
        for year, stars in sorted(self.year_stars.items()):
            n = sum(stars.values())
            yearly.append(dict(year=year, n=n, low_star=frequency(stars[1]+stars[2], n),
                               distribution=[dict(star=i, **frequency(stars[i], n)) for i in range(1, 6)]))
        valid_stars = sum(self.stars.values())
        valid_year = sum(row["n"] for row in yearly)
        availability = [dict(year=year, n=self.year_all[year],
                             body_nonblank=frequency(self.year_body[year],self.year_all[year]),
                             title_nonblank=frequency(self.year_title[year],self.year_all[year]),
                             verified_flag_valid=frequency(self.year_verified[year],self.year_all[year]),
                             image_url_checked=frequency(self.year_image_checked[year],self.year_all[year]),
                             review_image_url=frequency(self.year_review_image[year],self.year_image_checked[year]))
                        for year in sorted(self.year_all)]
        return dict(unit="parsed review object; duplicate copies retained",
                    valid_stars=frequency(valid_stars, raw_total), excluded_invalid_stars=raw_total-valid_stars,
                    by_star=by_star, valid_rating_and_time=frequency(valid_year, raw_total),
                    excluded_invalid_rating_or_time=raw_total-valid_year, by_year=yearly,
                    year_modality_availability=dict(
                        population='all parsed reviews with valid UTC time; rating need not be valid',
                        valid_timestamp=frequency(sum(self.year_all.values()),raw_total),
                        by_year=availability),
                    body_length_vs_helpful_vote=dict(overall=self.paired.result(raw_total),
                        by_star=[dict(star=i+1,**self.paired_by_star[i].result(self.stars[i+1]))
                                 for i in range(5)],
                        display_sample=self.pair_sample.result()))


class ProductVolume:
    def __init__(self):
        self.groups = defaultdict(Counter)
        self.equal_product_totals = defaultdict(float)

    def add(self, bucket, valid_ratings, low):
        group = self.groups[bucket]
        group["n"] += 1
        if valid_ratings:
            group["valid"] += 1
            group["stars"] += valid_ratings
            group["low"] += low
            self.equal_product_totals[bucket] += low / valid_ratings

    def result(self, buckets):
        return [dict(volume_bin=bucket, products=self.groups[bucket]["n"],
                     eligible_products=self.groups[bucket]["valid"],
                     excluded_products=self.groups[bucket]["n"]-self.groups[bucket]["valid"],
                     sum_product_low_star_fractions=self.equal_product_totals[bucket],
                     product_weighted_mean=rate(self.equal_product_totals[bucket], self.groups[bucket]["valid"]),
                     review_weighted_low_star=frequency(self.groups[bucket]["low"], self.groups[bucket]["stars"]))
                for bucket in buckets]
