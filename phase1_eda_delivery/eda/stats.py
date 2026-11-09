"""Bounded accumulators with explicit resolution and denominator semantics."""
import math
from collections import Counter


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def frequency(n, total):
    # Fail closed on invalid populations rather than publishing a plausible-looking
    # percentage. All counters in this EDA are exact, nonnegative integers.
    if (type(n) is not int or type(total) is not int
            or n < 0 or total < 0 or n > total):
        raise ValueError(f"E06 invalid frequency: numerator={n!r}, denominator={total!r}")
    return dict(numerator=n, denominator=total, fraction=rate(n, total))


class Distribution:
    """Exact frequencies on fixed grids. Log-bin quantiles are intervals."""
    def __init__(self, grid="log2", scale=1):
        self.grid = grid
        self.scale = scale
        self.bins = Counter()
        self.n = self.zero = 0
        self.minimum = self.maximum = None
        self.mean = self.m2 = 0.0
        self.m3 = self.m4 = 0.0
        self.moment_failure = False
        self.shape_failure = False
        self.on_value = None  # Optional disk-backed exact-frequency recorder.

    def add(self, x):
        if x is None:
            return
        if self.grid == "log2":
            if x == 0:
                key = -1075
            elif isinstance(x, int):
                key = x.bit_length() - 1
            else:
                key = math.frexp(x)[1] - 1
        elif self.grid == "linear_snapshot":
            key = min(16, math.floor((x - 1) * 4))
        else:
            key = x
        self.bins[key] += 1
        self.n += 1
        if self.on_value is not None:
            self.on_value(x)
        self.zero += x == 0
        self.minimum = x if self.minimum is None else min(self.minimum, x)
        self.maximum = x if self.maximum is None else max(self.maximum, x)
        if not self.moment_failure:
            try:
                value = x / self.scale
                delta = value - self.mean
                if not self.shape_failure and self.n > 1:
                    try:
                        old_m2,old_m3,old_m4 = self.m2,self.m3,self.m4
                        dn=delta/self.n
                        term=delta*dn*(self.n-1)
                        next_m4=(old_m4 + term*dn*dn*(self.n*self.n-3*self.n+3)
                                 + 6*dn*dn*old_m2-4*dn*old_m3)
                        next_m3=old_m3+term*dn*(self.n-2)-3*dn*old_m2
                        if not (math.isfinite(next_m3) and math.isfinite(next_m4)):
                            self.shape_failure=True
                        else:
                            self.m3, self.m4 = next_m3,next_m4
                    except (OverflowError,ZeroDivisionError):
                        self.shape_failure=True
                self.mean += delta / self.n
                self.m2 += delta * (value - self.mean)
                self.moment_failure = not math.isfinite(self.m2)
            except OverflowError:
                self.moment_failure = True

    def add_many(self, x, multiplicity):
        """Exact integer bins and stable two-population moment merge, O(1) per group.

        Required for millions of users, without iterating each ID again.
        """
        if type(multiplicity) is not int or multiplicity < 0:
            raise ValueError("multiplicity must be nonnegative integer")
        if not multiplicity:
            return
        if x is None:
            raise ValueError("weighted value cannot be null")
        if self.grid == "log2":
            key = (-1075 if x == 0 else x.bit_length()-1 if isinstance(x,int)
                   else math.frexp(x)[1]-1)
        elif self.grid == "linear_snapshot":
            key = min(16, math.floor((x-1)*4))
        else:
            key = x
        old_n = self.n
        if old_n or multiplicity != 1:
            # Weighted high moments not computed, rather than approximating from bins.
            self.shape_failure = True
        self.n += multiplicity
        self.bins[key] += multiplicity
        self.zero += multiplicity if x == 0 else 0
        self.minimum = x if self.minimum is None else min(self.minimum,x)
        self.maximum = x if self.maximum is None else max(self.maximum,x)
        if not self.moment_failure:
            try:
                v=x/self.scale
                delta=v-self.mean
                self.mean += delta*multiplicity/self.n
                self.m2 += delta*delta*old_n*multiplicity/self.n
                self.moment_failure = not math.isfinite(self.m2)
            except (OverflowError, ZeroDivisionError):
                self.moment_failure=True

    def bounds(self, key):
        if self.grid == "log2":
            if key == -1075:
                return 0, 0
            return 2 ** key, 2 ** (key + 1)
        if self.grid == "linear_snapshot":
            return 1 + key / 4, min(5, 1 + (key + 1) / 4)
        return key / self.scale, key / self.scale

    def summary(self, total):
        if self.n != sum(self.bins.values()) or self.zero > self.n or total < self.n:
            raise ValueError("E31 distribution count conservation failed")
        bins = []
        quantiles = {}
        targets = iter((name, max(1, (self.n * num + 99) // 100))
                       for name, num in (("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)))
        target = next(targets, None)
        cumulative = 0
        for key, count in sorted(self.bins.items()):
            lo, hi = self.bounds(key)
            bins.append(dict(lower=lo, upper=hi, **frequency(count, self.n)))
            cumulative += count
            while target and cumulative >= target[1]:
                quantiles[target[0]] = [lo, hi]
                target = next(targets, None)
        if self.n:
            q1, q3 = quantiles["p25"], quantiles["p75"]
            try:
                iqr = [max(0, q3[0] - q1[1]), q3[1] - q1[0]]
            except OverflowError:
                iqr = None
        else:
            iqr = None
        skewness=kurtosis=None
        if self.n and not self.moment_failure and not self.shape_failure and self.m2 > 0:
            try:
                m2=self.m2/self.n
                skewness=(self.m3/self.n)/(m2**1.5)
                kurtosis=(self.m4/self.n)/(m2*m2)-3
                if not math.isfinite(skewness) or not math.isfinite(kurtosis):
                    skewness=kurtosis=None
            except (OverflowError,ZeroDivisionError):
                pass
        # Conservative tail-count envelope relative to a binned P95 threshold.
        # It is an interval, never a claim of exact outlier count.
        tail_lower=tail_upper=0
        if self.n:
            tlo,thi=quantiles['p95']
            for item in bins:
                lo,hi=item['lower'],item['upper']
                if lo>=thi and not (lo==hi==thi):
                    tail_lower+=item['numerator']
                if hi>tlo:
                    tail_upper+=item['numerator']
        most_frequent=max((x['numerator'] for x in bins),default=0)
        return dict(total=total, valid=self.n, excluded=total-self.n, bins=bins,
                    bin_rule=self.grid, zero=frequency(self.zero, self.n),
                    quantile_method="inverse empirical CDF (nearest rank); brackets are bin bounds",
                    quantile_bounds=quantiles, iqr_bounds=iqr,
                    minimum=None if self.minimum is None else self.minimum if self.scale == 1 else self.minimum / self.scale,
                    maximum=None if self.maximum is None else self.maximum if self.scale == 1 else self.maximum / self.scale,
                    mean=self.mean if self.n and not self.moment_failure else None,
                    population_sd=math.sqrt(max(0, self.m2) / self.n) if self.n and not self.moment_failure else None,
                    moment_status="outside floating point range" if self.moment_failure else "ok",
                    skewness_population=skewness,excess_kurtosis_population=kurtosis,
                    modal_bin_count=most_frequent,
                    above_p95_count_bounds=[tail_lower,tail_upper] if self.n else None,
                    above_p95_bounds_definition="lower=definitely beyond P95 bin; upper=possibly beyond P95 bin; no clipping",
                    shape_status="not_estimable_or_nonfinite" if skewness is None else "computed_from_streaming_full_values")


def text_state(row, field):
    if field not in row:
        return "absent"
    value = row[field]
    if value is None:
        return "null"
    if not isinstance(value, str):
        return "wrong_type"
    return "nonblank" if value.strip() else "blank"


def state_table(counts, total, levels):
    return [dict(level=level, **frequency(counts[level], total)) for level in levels]


def volume_bin(n):
    return ("1–4" if n < 5 else "5–14" if n < 15 else "15–49" if n < 50
            else "50–199" if n < 200 else "200+")
