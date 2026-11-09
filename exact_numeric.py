"""E09/E26/E27: disk-backed exact frequency summaries; never export raw row IDs.

Conserves the validated numeric observations without retaining their values in
RAM. SQLite REAL is injective for finite Python floats, and integers whose
magnitude is <= 2**53; larger integers are deliberately refused *for exact
quantiles* rather than silently rounded. Legacy binned statistics are retained.
"""
from collections import Counter
import math

MAX_EXACT_INT = 1 << 53
PERCENTILES = (('p05', 5, 100), ('p25', 25, 100), ('p50', 50, 100), ('p75', 75, 100),
               ('p95', 95, 100), ('p99', 99, 100), ('p999', 999, 1000))


class ExactNumeric:
    MAX_PENDING_KEYS = 6000

    def __init__(self, database):
        self.db = database
        self.db.execute('''CREATE TABLE exact_numeric (
            category TEXT NOT NULL, role TEXT NOT NULL, metric TEXT NOT NULL,
            value REAL NOT NULL, count INTEGER NOT NULL,
            PRIMARY KEY (category, role, metric, value)
        ) WITHOUT ROWID''')
        self.pending = Counter()
        self.rejected = Counter()

    def add(self, category, role, metric, value):
        tag = (category, role, metric)
        if type(value) not in (int, float):
            self.rejected[(tag, 'wrong_type')] += 1
            return
        try:
            if type(value) is int and abs(value) > MAX_EXACT_INT:
                self.rejected[(tag, 'integer_exceeds_exact_binary64_range')] += 1
                return
            converted = float(value)
            if not math.isfinite(converted):
                self.rejected[(tag, 'nonfinite_or_out_of_range')] += 1
                return
        except (OverflowError, ValueError):
            self.rejected[(tag, 'nonfinite_or_out_of_range')] += 1
            return
        self.pending[(*tag, converted)] += 1
        if len(self.pending) >= self.MAX_PENDING_KEYS:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.db.executemany('''INSERT INTO exact_numeric VALUES (?,?,?,?,?)
            ON CONFLICT(category,role,metric,value)
            DO UPDATE SET count=count+excluded.count''',
            ((category, role, metric, value, count)
             for (category,role,metric,value),count in self.pending.items()))
        self.pending.clear()

    def count(self, category, role, metric):
        self.flush()
        return self.db.execute('''SELECT COALESCE(SUM(count),0) FROM exact_numeric
            WHERE category=? AND role=? AND metric=?''',
            (category,role,metric)).fetchone()[0] + sum(
                n for ((tag,reason),n) in self.rejected.items()
                if tag==(category,role,metric))

    def total_variation(self, category, before, after, metric, nr, nu):
        """Exact categorical empirical TV using a disk-side GROUP BY."""
        if not nr or not nu:
            return None
        for role in (before, after):
            if any(tag == (category,role,metric) for tag,reason in self.rejected):
                return None  # A partial histogram must never be called exact TV.
        self.flush()
        query = """SELECT SUM(ABS(raw_n*1.0/? - unique_n*1.0/?))/2.0
                FROM (SELECT value,
                  SUM(CASE WHEN role=? THEN count ELSE 0 END) AS raw_n,
                  SUM(CASE WHEN role=? THEN count ELSE 0 END) AS unique_n
                FROM exact_numeric WHERE category=? AND metric=?
                 AND role IN (?,?) GROUP BY value)"""
        return self.db.execute(query,(nr,nu,before,after,category,metric,before,after)).fetchone()[0]

    def profile(self, category, role, metric, expected):
        """Nearest-rank quantiles and Tukey 1.5*IQR fences (all exact counts)."""
        if type(expected) is not int or expected < 0:
            raise ValueError('Expected count must be a nonnegative integer')
        self.flush()
        tag = (category, role, metric)
        rejected = {reason: count for (prefix,reason),count in self.rejected.items()
                    if prefix == tag}
        n = self.db.execute('''SELECT COALESCE(SUM(count),0) FROM exact_numeric
            WHERE category=? AND role=? AND metric=?''', tag).fetchone()[0]
        if n + sum(rejected.values()) != expected:
            raise ValueError(f'E26 exact-frequency conservation failed for {tag}: '
                             f'{n}+{sum(rejected.values())}!={expected}')
        base = dict(category=category, role=role, metric=metric,
                    expected_valid_observations=expected, frequency_total=n,
                    rejected_for_exact=rejected,
                    unit='one validated numeric observation in the specified scope',
                    quantile_definition='nearest rank: sorted observation ceil(p*n), 1-indexed',
                    outlier_definition='Tukey fences Q1-1.5*IQR, Q3+1.5*IQR; strict outside; descriptive, never auto-delete')
        if rejected:
            return dict(base, status='not_exact_unrepresentable_values', quantiles=None,
                        iqr=None, tukey=None)
        if not n:
            return dict(base, status='no_valid_observations', quantiles=None,
                        iqr=None, tukey=None)
        ranks = {name: max(1, (n * numerator + denominator - 1) // denominator)
                 for name,numerator,denominator in PERCENTILES}
        vals = {}
        seen = 0
        for value, count in self.db.execute('''SELECT value,count FROM exact_numeric
                WHERE category=? AND role=? AND metric=? ORDER BY value''', tag):
            seen += count
            for name,rank in ranks.items():
                if name not in vals and seen >= rank:
                    vals[name] = value
        if seen != n or len(vals) != len(ranks):
            raise ValueError('E26 sorted-frequency rank conservation failure')
        q1, q3 = vals['p25'], vals['p75']
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if not all(math.isfinite(x) for x in (iqr, lower, upper)):
            return dict(base, status='exact_quantiles_fence_overflow',
                        quantiles=vals, iqr=iqr if math.isfinite(iqr) else None,
                        tukey=None)
        below, above = self.db.execute('''SELECT
                COALESCE(SUM(CASE WHEN value < ? THEN count ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN value > ? THEN count ELSE 0 END),0)
                FROM exact_numeric WHERE category=? AND role=? AND metric=?''',
                (lower,upper,*tag)).fetchone()
        return dict(base, status='exact_disk_frequency', quantiles=vals, iqr=iqr,
                    tukey=dict(lower_fence=lower,upper_fence=upper,
                               below_count=below,above_count=above,total_outlier_count=below+above,
                               denominator=n, fraction=(below+above)/n),
                    exact_distinct_values=self.db.execute('''SELECT COUNT(*) FROM exact_numeric
                        WHERE category=? AND role=? AND metric=?''',tag).fetchone()[0])


def review_core_metrics(row):
    """Subset of core-eligible review measures (do not confuse with all raw rows)."""
    from quality_audit import integer
    votes = row.get('helpful_vote')
    if integer(votes) and votes >= 0:
        yield 'helpful_vote', votes
    body = row.get('text')
    if isinstance(body, str):
        yield 'body_chars_all_strings', len(body)
        yield 'body_words_all_strings', len(body.split())
    title = row.get('title')
    if isinstance(title, str):
        yield 'title_chars_all_strings', len(title)
    images = row.get('images')
    if isinstance(images, list):
        yield 'review_image_entries', len(images)


def metadata_metrics(row):
    from quality_audit import number, integer
    price = row.get('price')
    if number(price) and price >= 0:
        yield 'price', price
    rating = row.get('average_rating')
    if number(rating) and 1 <= rating <= 5:
        yield 'average_rating', rating
    count = row.get('rating_number')
    if integer(count) and count >= 0:
        yield 'rating_number', count
    images = row.get('images')
    if isinstance(images, list):
        yield 'images_entries', len(images)
