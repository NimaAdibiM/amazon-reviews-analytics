"""Category-product volumes and globally reconciled provisional 5+10 cohorts."""
from collections import Counter, defaultdict
from hashlib import sha256
from heapq import heappush, heapreplace
from itertools import combinations, groupby

import structure_audit as s

from .multivariate import ProductVolume
from .stats import Distribution, frequency, rate, volume_bin


VOLUMES = ("1–4", "5–14", "15–49", "50–199", "200+")
JOIN_STATES = ("no_metadata_key", "ambiguous_metadata", "one_canonical_version")
COHORT_STATES = ("no_valid_review", "one_to_four", "five_to_fourteen", "tie_at_5_only",
                 "tie_at_15_only", "both_boundary_ties", "conflict_excluded", "eligible")
POLICIES = ("exclude", "retain")
FIVE, FIFTEEN, SIXTEEN = 5, 15, 16


def raw_product_results(index, category, raw_total):
    raw_volume, core_volume = Distribution(), Distribution()
    volumes = Counter()
    join_products, join_reviews, joined = Counter(), Counter(), Counter()
    relation = ProductVolume()
    # Snapshot metadata groups use ONLY products with a unique canonical metadata
    # version. Equal-product and review-weighted estimates are never conflated.
    conditioned = {facet:{state:Counter() for state in ('available','unavailable')}
                   for facet in ('valid_price','product_image_url','descriptive_content')}
    # E38: mutually exclusive price buckets, by product, for *one canonical*
    # metadata version only. Raw review counts are never used as product weights.
    price_bands = ('0','(0,25]','(25,100]','(100,500]','(500,+inf)')
    by_price = {bucket: Counter() for bucket in price_bands}
    for n, stars, low, core, versions, _, image, price, content, price_band in index.raw_products(category):
        raw_volume.add(n)
        core_volume.add(core)
        bucket = volume_bin(n)
        volumes[bucket] += 1
        relation.add(bucket, stars, low)
        join = JOIN_STATES[0 if versions is None else 1 if versions > 1 else 2]
        join_products[join] += 1
        join_reviews[join] += n
        if versions == 1:
            if price_band is not None:
                if price_band not in by_price or not price:
                    raise ValueError('E38 inconsistent valid-price bucket')
                b = by_price[price_band]
                b['matched_products'] += 1
                b['raw_reviews'] += n
                if stars:
                    b['rated_products'] += 1
                    b['rated_reviews'] += stars
                    b['low_reviews'] += low
                    b['sum_product_low_fraction'] += low/stars
            for facet, flag in (("valid_price",price), ("product_image_url",image),
                                ("descriptive_content",content)):
                counter = conditioned[facet]['available' if flag else 'unavailable']
                counter['all_products'] += 1
                counter['raw_reviews'] += n
                if stars:
                    counter['rated_products'] += 1
                    counter['rated_reviews'] += stars
                    counter['low_reviews'] += low
                    counter['sum_product_low_fraction'] += low/stars
            for label, flag in (("product_image_url", image), ("valid_price", price),
                                ("descriptive_content", content)):
                joined[label] += n * flag
    indexed_reviews = sum(join_reviews.values())
    metadata_keys, metadata_only, ambiguous = index.metadata_keys(category)
    review_match_denominator = join_reviews[JOIN_STATES[2]]
    return dict(unit="product (raw parent_asin) within one review file",
                total=raw_volume.n, raw_review_volume=raw_volume.summary(raw_volume.n),
                valid_core_review_volume=core_volume.summary(raw_volume.n),
                raw_volume_groups={k: frequency(volumes[k], raw_volume.n) for k in VOLUMES},
                volume_vs_rating=relation.result(VOLUMES),
                metadata_conditioned_ratings=dict(
                    population='Within-category raw parent products with a UNIQUE canonical metadata version; observational snapshot',
                    warning='Metadata is crawl-time snapshot, not historical data available at fifth review; no predictive use',
                    price_band_analysis=dict(
                        status='descriptive_snapshot_not_historical',
                        source='numeric nonnegative metadata price, unadjusted; bucket ranges in raw supplied units',
                        population='review products with exactly one canonical metadata version and valid numeric price',
                        unmatched_or_ambiguous_products_excluded=True,
                        bands=[dict(price_band=bucket, matched_products=int(by_price[bucket]['matched_products']),
                                    raw_reviews_of_matched_products=int(by_price[bucket]['raw_reviews']),
                                    products_with_valid_ratings=frequency(int(by_price[bucket]['rated_products']),int(by_price[bucket]['matched_products'])),
                                    low_star_equal_product_mean=rate(by_price[bucket]['sum_product_low_fraction'],by_price[bucket]['rated_products']),
                                    low_star_review_weighted=frequency(int(by_price[bucket]['low_reviews']),int(by_price[bucket]['rated_reviews'])))
                               for bucket in price_bands],
                        note='No causal, currency, inflation, time-of-review or available-at-t5 claims'),
                    facets={facet:{state:dict(
                        all_matched_products=int(c['all_products']),
                        products_with_valid_ratings=frequency(int(c['rated_products']),int(c['all_products'])),
                        raw_reviews_in_matched_products=int(c['raw_reviews']),
                        low_star_product_weighted_mean=rate(c['sum_product_low_fraction'],c['rated_products']),
                        low_star_review_weighted=frequency(int(c['low_reviews']),int(c['rated_reviews'])))
                        for state,c in states.items()} for facet,states in conditioned.items()}),
                metadata_connection=dict(
                    unit="one raw review object or one review product within this file pair, as labelled",
                    review_states={**{k: frequency(join_reviews[k], raw_total) for k in JOIN_STATES},
                                   "unusable_parent": frequency(raw_total-indexed_reviews, raw_total)},
                    product_states={k: frequency(join_products[k], raw_volume.n) for k in JOIN_STATES},
                    modalities_among_unique_matches={k: frequency(joined[k], review_match_denominator)
                                                     for k in ("product_image_url", "valid_price", "descriptive_content")},
                    metadata_key_count=metadata_keys, metadata_only_keys=frequency(metadata_only, metadata_keys),
                    ambiguous_metadata_keys=frequency(ambiguous, metadata_keys),
                    extra_canonical_copies_for_valid_metadata_keys=index.meta_copies[category]))


class BottomK:
    """Hash-based product sampling without replacement; no identifiers exported."""
    def __init__(self, k, seed):
        self.k, self.seed = k, seed
        self.heap = []

    def add(self, parent, point):
        if not self.k:
            return
        key = int.from_bytes(sha256(f"{self.seed}\0{parent}".encode("utf-8")).digest(), "big")
        entry = (-key, parent, point)
        if len(self.heap) < self.k:
            heappush(self.heap, entry)
        elif entry > self.heap[0]:
            heapreplace(self.heap, entry)

    def result(self, total):
        return dict(method="lowest SHA-256 ranks of seed + NUL + raw parent; product without replacement",
                    population="eligible products in this policy and primary category",
                    population_size=total, size=len(self.heap), cap=self.k, seed=self.seed,
                    nominal_inclusion_probability=rate(len(self.heap), total) if self.k else None,
                    points=[point for _, _, point in sorted(self.heap, reverse=True)])


class Cohort:
    def __init__(self, sample_size, seed):
        self.n = Counter()
        self.status = Counter()
        self.early = defaultdict(Counter)
        self.maturity = Distribution()
        self.early_span_by_outcome = defaultdict(Distribution)
        self.early_low_hist_by_outcome = defaultdict(Counter)
        self.early_text_hist_by_outcome = defaultdict(Counter)
        self.t15_years = Counter()
        self.no_overlap_target = Distribution("discrete", 10)
        self.no_overlap_maturity = Distribution()
        self.target = Distribution("discrete", 10)
        # E44/E50: optional *alternative* conflict rule, keeping the strict
        # whole-observed-sequence exclusion as the default primary policy.
        self.window15_target = Distribution("discrete", 10)
        self.target_by_t5_year = defaultdict(lambda: Distribution("discrete", 10))
        self.target_by_early = defaultdict(Counter)
        self.maturity_by_t5_year = defaultdict(Distribution)
        self.early_years = defaultdict(Counter)
        self.sample = BottomK(sample_size, seed)

    def add(self, policy, parent, n, first, conflicts, multi, units_per_day,
            window15_conflicts=None):
        # A count is only available if full candidate-event identifiers were
        # supplied by the disk index. Direct unit fixtures with five-field
        # records do not silently imply zero in-window conflicts.
        if window15_conflicts is not None:
            if not (type(window15_conflicts) is int and 0 <= window15_conflicts <= conflicts):
                raise ValueError('E44 invalid in-window conflict count')
            self.n['window15_audited_products'] += 1
            self.n['window15_candidate_groups'] += window15_conflicts
            self.n['candidate_groups_not_repeated_in_window15'] += conflicts-window15_conflicts
        self.n["products"] += 1
        self.n["multi_category_products"] += multi
        self.n["has_conflict_candidate"] += conflicts > 0
        self.n["candidate_event_groups"] += conflicts
        self.n["exact_dedup_core_reviews"] += n
        self.n["any_valid"] += n >= 1
        self.n["at_least_five"] += n >= FIVE
        self.n["at_least_fifteen"] += n >= FIFTEEN
        tie5 = n >= 6 and first[4][1] == first[5][1]
        tie15 = n >= SIXTEEN and first[14][1] == first[15][1]
        self.n["tie5_among_at_least_five"] += tie5
        self.n["tie5_among_at_least_fifteen"] += n >= FIFTEEN and tie5
        self.n["tie15_among_at_least_fifteen"] += tie15
        self.n["both_ties_among_at_least_fifteen"] += n >= FIFTEEN and tie5 and tie15
        if n >= FIFTEEN and not tie5 and not tie15:
            self.n["clear_boundaries"] += 1
        if not n:
            status = "no_valid_review"
        elif n < FIVE:
            status = "one_to_four"
        elif n < FIFTEEN:
            status = "five_to_fourteen"
        elif tie5 or tie15:
            status = ("both_boundary_ties" if tie5 and tie15 else
                      "tie_at_5_only" if tie5 else "tie_at_15_only")
        elif policy == "exclude" and conflicts:
            status = "conflict_excluded"
        else:
            status = "eligible"
        self.status[status] += 1
        if n < FIVE or tie5:
            return
        early_low = sum(item[2] <= 2 for item in first[:FIVE])
        early_text = sum(item[3] for item in first[:FIVE])
        group = self.early[status]
        group["products"] += 1
        group["low"] += early_low
        group["text"] += early_text
        group["star_sum"] += sum(item[2] for item in first[:FIVE])
        self.early_low_hist_by_outcome[status][early_low] += 1
        self.early_text_hist_by_outcome[status][early_text] += 1
        first_five_span = (int(first[4][1]) - int(first[0][1])) / units_per_day
        if first_five_span < 0:
            raise AssertionError("First-five sequence is not time-ordered")
        self.early_span_by_outcome[status].add(first_five_span)
        t5_year = first[4][4]
        self.early_years[status][t5_year] += 1
        # E44: the alternative rule disregards candidate conflicts *entirely*
        # outside the observed first 15, but never resolves a conflict by
        # choosing a desired rating. Only report it when records are audited.
        if (window15_conflicts is not None and n >= FIFTEEN and
                not tie5 and not tie15 and window15_conflicts == 0):
            self.n['window15_eligible'] += 1
            self.window15_target.add(sum(item[2] <= 2 for item in first[FIVE:FIFTEEN]))
            if conflicts:
                self.n['window15_added_vs_global_exclude'] += 1
        if status != "eligible":
            return
        t5, t15 = (int(first[i][1]) for i in (4, 14))
        lag = (t15 - t5) / units_per_day
        future_low = sum(item[2] <= 2 for item in first[FIVE:FIFTEEN])
        self.n["eligible_without_overlap"] += not multi
        self.n["early_windows_any_text"] += early_text > 0
        if lag < 0:
            raise AssertionError("Negative outcome maturity lag")
        self.target.add(future_low)
        self.maturity.add(lag)
        self.t15_years[first[14][4]] += 1
        if not multi:
            self.no_overlap_target.add(future_low)
            self.no_overlap_maturity.add(lag)
        self.maturity_by_t5_year[t5_year].add(lag)
        self.target_by_t5_year[t5_year].add(future_low)
        self.target_by_early[early_low][future_low] += 1
        self.sample.add(parent, dict(sequence_reviews=n, maturity_days=lag))

    def result(self):
        n = self.n
        eligible = self.status["eligible"]
        stages = (("valid_product_id", n["products"], n["products"]),
                  ("has_core", n["any_valid"], n["products"]),
                  ("at_least_five_exact_dedup", n["at_least_five"], n["any_valid"]),
                  ("at_least_fifteen_exact_dedup", n["at_least_fifteen"], n["at_least_five"]),
                  ("unambiguous_boundaries", n["clear_boundaries"], n["at_least_fifteen"]),
                  ("eligible_under_policy", eligible, n["clear_boundaries"]))
        early = {}
        for key in COHORT_STATES:
            row = self.early[key]
            products = row["products"]
            early[key] = dict(products=products, excluded_no_unambiguous_first_five=self.status[key]-products,
                              low_star=frequency(row["low"], FIVE*products),
                              any_text=frequency(row["text"], FIVE*products),
                              mean_rating=dict(sum_stars=row["star_sum"], denominator=FIVE*products,
                                               value=rate(row["star_sum"], FIVE*products)),
                              first_five_span_days=self.early_span_by_outcome[key].summary(products),
                              first_five_low_counts=[dict(count=i,**frequency(self.early_low_hist_by_outcome[key][i], products))
                                                     for i in range(6)],
                              first_five_nonblank_text_counts=[dict(count=i,**frequency(self.early_text_hist_by_outcome[key][i], products))
                                                               for i in range(6)],
                              t5_year=[dict(year=year, **frequency(count, products))
                                       for year, count in sorted(self.early_years[key].items())])
        return dict(unit="global unique raw parent_asin assigned once to primary review-file category",
                    total=n["products"], deduplicated_core_review_records=n["exact_dedup_core_reviews"],
                    has_candidate_event_conflict=frequency(n["has_conflict_candidate"], n["products"]),
                    candidate_event_groups=n["candidate_event_groups"],
                    funnel=[dict(stage=key, excluded=den-num, **frequency(num, den),
                                 fraction_of_id_products=rate(num, n["products"])) for key, num, den in stages],
                    mutually_exclusive_outcomes={key: frequency(self.status[key], n["products"])
                                                 for key in COHORT_STATES},
                    boundary_ties=dict(t5_t6_from_5=frequency(n["tie5_among_at_least_five"], n["at_least_five"]),
                                       t5_t6_from_15=frequency(n["tie5_among_at_least_fifteen"], n["at_least_fifteen"]),
                                       t15_t16=frequency(n["tie15_among_at_least_fifteen"], n["at_least_fifteen"]),
                                       both=frequency(n["both_ties_among_at_least_fifteen"], n["at_least_fifteen"])),
                    multi_category_products=frequency(n["multi_category_products"], n["products"]),
                    eligible_without_overlap=frequency(n["eligible_without_overlap"],
                                                       n["products"]-n["multi_category_products"]),
                    early_windows_with_any_text=frequency(n["early_windows_any_text"], eligible),
                    early_by_outcome=early,
                    # E50: the without-overlap analysis keeps the *same eligible*
                    # product definition, changing only the multi-category filter.
                    no_overlap_sensitivity=dict(
                        included_products=n["eligible_without_overlap"],
                        excluded_overlap_products=eligible-n["eligible_without_overlap"],
                        future_low_share=self.no_overlap_target.summary(n["eligible_without_overlap"]),
                        maturity_days=self.no_overlap_maturity.summary(n["eligible_without_overlap"]),
                        limitation="Selection sensitivity, not a causal effect of category overlap"),
                    t15_year=[dict(year=year,**frequency(count,eligible))
                              for year,count in sorted(self.t15_years.items())],
                    future_low_share=self.target.summary(eligible),
                    window15_conflict_policy_sensitivity=dict(
                        status='computed' if n['window15_audited_products']==n['products'] else 'not_audited_for_all_products',
                        candidate_conflicts_in_first15=n['window15_candidate_groups'],
                        candidate_groups_not_repeated_in_first15=n['candidate_groups_not_repeated_in_window15'],
                        eligible_products_if_only_first15_conflicts_excluded=n['window15_eligible'] if n['window15_audited_products']==n['products'] else None,
                        newly_included_vs_global_exclude=n['window15_added_vs_global_exclude'] if n['window15_audited_products']==n['products'] else None,
                        future_low_share=self.window15_target.summary(n['window15_eligible']) if n['window15_audited_products']==n['products'] else None,
                        note='EXPLORATORY selection sensitivity only. Main exclude still excludes all candidate conflicts in observed sequence; boundary ties remain excluded. A matching user and timestamp is not a verified event identity.'),
                    target_by_t5_year={str(year): stat.summary(stat.n)
                                       for year, stat in sorted(self.target_by_t5_year.items())},
                    maturity_days=self.maturity.summary(eligible),
                    maturity_by_t5_year={str(year): stat.summary(stat.n)
                                         for year, stat in sorted(self.maturity_by_t5_year.items())},
                    early_future_joint=[dict(early_low_count=k, future_low_count=j,
                                             **frequency(self.target_by_early[k][j],
                                                         sum(self.target_by_early[k].values())))
                                        for k in range(6) for j in range(11)],
                    optional_product_sample=self.sample.result(eligible))


def reconcile(index, category_file_names, policies, sample_size, seed, unit, core_star_counts_raw=None):
    sets = {policy: {cat: Cohort(sample_size if i == 0 else 0, seed)
                     for cat in category_file_names}
            for i, policy in enumerate(policies)}
    reviews = iter(groupby(index.reviews(), key=lambda row: row[0])) if index.sequence_enabled else iter(())
    current, entries = next(reviews, (None, ()))
    conflicts = iter(index.conflicts()) if index.sequence_enabled else iter(())
    conflict_parent, conflict_groups, _ = next(conflicts, (None, 0, 0))
    patterns, pairs, memberships, eligible_by_membership, primary = (Counter() for _ in range(5))
    core_pre_dedup = unique_core = 0
    units_per_day = 86400 * s.TIMESTAMP_UNITS[unit]
    for parent, rows in groupby(index.memberships(), key=lambda row: row[0]):
        group = list(rows)  # at most four memberships for one parent
        cats = tuple(row[1] for row in group)
        main = min(cats, key=lambda cat: (category_file_names[cat], cat))
        primary[main] += 1
        patterns[cats] += 1
        pairs.update(combinations(cats, 2))
        memberships.update(cats)
        core_pre_dedup += sum(row[2] for row in group)
        if not index.sequence_enabled:
            continue
        n, first = 0, []
        if current == parent:
            for n, item in enumerate(entries, 1):
                if n <= SIXTEEN:
                    first.append(item)
            current, entries = next(reviews, (None, ()))
        unique_core += n
        groups = 0
        if conflict_parent == parent:
            groups = conflict_groups
            conflict_parent, conflict_groups, _ = next(conflicts, (None, 0, 0))
        # The sixth element is the canonical usable user key. Within-window
        # candidate counts require two distinct canonical review objects with
        # the same (user,time) among first fifteen. They are NOT a proof that
        # the records represent one human review event.
        candidate_pairs = Counter((row[5],row[1]) for row in first[:FIFTEEN]
                                  if row[5] is not None)
        in_window = sum(count > 1 for count in candidate_pairs.values())
        if in_window > groups:
            raise RuntimeError('E44 window candidate groups exceed global candidates')
        for policy, cohorts in sets.items():
            cohort = cohorts[main]
            before = cohort.status["eligible"]
            cohort.add(policy, parent, n, first, groups, len(cats) > 1, units_per_day,
                       window15_conflicts=in_window)
            if policy == policies[0] and cohort.status["eligible"] > before:
                eligible_by_membership.update(cats)
    if current is not None or conflict_parent is not None:
        raise RuntimeError("Global stream did not align with indexed product membership.")
    total = sum(patterns.values())
    global_summary = dict(
        unit="global unique raw parent_asin across parsed review files; metadata-only IDs excluded",
        total_unique_products=total, total_category_product_memberships=sum(memberships.values()),
        multiple_categories=frequency(sum(v for k, v in patterns.items() if len(k) > 1), total),
        category_memberships={cat: frequency(memberships[cat], total) for cat in category_file_names},
        primary_categories={cat: frequency(primary[cat], total) for cat in category_file_names},
        primary_rule="ascending review filename (case-sensitive); category name breaks an identical basename",
        membership_patterns=[dict(categories=list(k), **frequency(v, total)) for k, v in sorted(patterns.items())],
        pairwise_overlaps=[dict(categories=list(pair),
                                **frequency(pairs[pair], memberships[pair[0]]+memberships[pair[1]]-pairs[pair]))
                           for pair in combinations(sorted(category_file_names), 2)],
        core_records_before_exact_dedup=core_pre_dedup,
        core_records_after_exact_dedup=unique_core if index.sequence_enabled else None,
        removed_exact_core_copies=frequency(core_pre_dedup-unique_core, core_pre_dedup) if index.sequence_enabled else None,
        extra_core_copies_by_source_file=dict(index.review_copies) if index.sequence_enabled else None,
        eligible_products_by_original_membership={cat: frequency(eligible_by_membership[cat], memberships[cat])
                                                  for cat in category_file_names} if index.sequence_enabled else None)
    if index.sequence_enabled and core_pre_dedup-unique_core != sum(index.review_copies.values()):
        raise RuntimeError("Global exact-copy ledger failed to reconcile.")
    if core_star_counts_raw is not None and index.sequence_enabled:
        before = {str(i): sum(core_star_counts_raw[cat][str(i)] for cat in category_file_names)
                  for i in range(1, 6)}
        after = index.unique_core_star_counts()
        if sum(before.values()) != core_pre_dedup or sum(after.values()) != unique_core:
            raise RuntimeError("E09 core rating sensitivity failed to conserve raw/unique totals")
        pre_low, post_low = before['1'] + before['2'], after['1'] + after['2']
        total_variation = (sum(abs(before[str(i)] / core_pre_dedup - after[str(i)] / unique_core)
                              for i in range(1, 6)) / 2 if core_pre_dedup and unique_core else None)
        global_summary['core_rating_exact_duplicate_sensitivity'] = {
            'status': 'computed_for_full_core_global_population_only',
            'population': 'all valid-parent, valid-star, representable-time raw review objects across four files',
            'dedup_definition': 'full canonical parsed record equality, across category files',
            'raw_core_stars': before, 'exact_unique_core_stars': after,
            'raw_core_n': core_pre_dedup, 'exact_unique_core_n': unique_core,
            'raw_low_star_1_2': frequency(pre_low, core_pre_dedup),
            'exact_unique_low_star_1_2': frequency(post_low, unique_core),
            'total_variation_5_star_fractions': total_variation,
            'limitation': 'This is NOT the raw all-review distribution, nor a within-category unique estimate; it cannot replace the product-level 5+10 target.'}
    result = {policy: {cat: cohort.result() for cat, cohort in cohorts.items()}
              for policy, cohorts in sets.items()} if index.sequence_enabled else {}
    return global_summary, result
