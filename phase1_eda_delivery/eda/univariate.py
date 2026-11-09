"""One-variable summaries over the parsed, pre-cleaning record populations."""
from collections import Counter, defaultdict
import re
import math

import quality_audit as q
import structure_audit as s

from .stats import Distribution, frequency, state_table, text_state


TEXT_STATES = ("absent", "null", "wrong_type", "blank", "nonblank")
REVIEW_MODALITIES = ("rating", "timestamp", "title", "body", "any_text", "review_image_url",
                     "parent", "interaction_keys", "helpful_vote", "verified_purchase")
META_MODALITIES = ("valid_price", "average_rating", "rating_number", "product_image_url",
                   "parent", "descriptive_content")
SURROGATE = re.compile(r"[\ud800-\udfff]")
BODY_FLAGS = ("at_most_3_words", "over_10000_codepoints", "non_ascii",
              "replacement_character", "surrogate_codepoint")
REVIEW_METRICS = {
    "rating": ("discrete", "stars"),
    "title_chars": ("log2", "Unicode code points"),
    "body_chars": ("log2", "Unicode code points"),
    "body_words": ("log2", "whitespace-delimited tokens"),
    "helpful_vote": ("log2", "votes"),
    # Added E16/E24 views. String-valued fields include blank/whitespace strings;
    # the legacy nonblank-only series above remain unchanged and clearly labelled.
    "title_chars_all_strings": ("log2", "Unicode code points; all string values"),
    "title_words_all_strings": ("log2", "whitespace-delimited tokens; all string values"),
    "body_chars_all_strings": ("log2", "Unicode code points; all string values"),
    "body_words_all_strings": ("log2", "whitespace-delimited tokens; all string values"),
    "review_image_entries": ("log2", "top-level list entries (not verified images)"),
}
META_METRICS = {
    "price": ("log2", "price as supplied"),
    "average_rating": ("linear_snapshot", "stars"),
    "rating_number": ("log2", "ratings"),
    "title_chars_all_strings": ("log2", "Unicode code points; all string values"),
    "features_entries": ("log2", "top-level list entries"),
    "features_chars": ("log2", "Unicode code points in all-string lists"),
    "description_entries": ("log2", "top-level list entries"),
    "description_chars": ("log2", "Unicode code points in all-string lists"),
    "images_entries": ("log2", "top-level list entries (not distinct images)"),
    "videos_entries": ("log2", "top-level list entries"),
    "categories_entries": ("log2", "top-level list entries"),
    "bought_together_entries": ("log2", "top-level list entries"),
    "details_keys": ("log2", "dictionary keys"),
}


class RawCategory:
    def __init__(self, timestamp_unit):
        self.unit = timestamp_unit
        self.review_stats = {name: Distribution(grid) for name, (grid, _) in REVIEW_METRICS.items()}
        self.meta_stats = {name: Distribution(grid) for name, (grid, _) in META_METRICS.items()}
        self.review_modalities = Counter(dict.fromkeys(REVIEW_MODALITIES, 0))
        self.meta_modalities = Counter(dict.fromkeys(META_MODALITIES, 0))
        self.text_states = {"title": Counter(), "text": Counter()}
        self.years = Counter()
        self.months = Counter()
        self.verified_states = Counter()
        self.body_flags = Counter()
        self.core_masks = Counter()
        self.core_stars = Counter()
        self.min_ts = self.max_ts = None
        self.review_field_states = defaultdict(Counter)
        self.meta_field_states = defaultdict(Counter)
        self.category_path_states = Counter()
        self.script_family_records = Counter()
        self.script_family_letters = Counter()
        self.character_anomalies = Counter()
        self.string_body_records = 0
        self.reviews = self.metadata = 0

    def bind_exact_numeric(self, recorder, category):
        """Stream every accepted EDA metric into bounded disk frequencies.

        The original Distribution summaries and all their chart contracts remain
        unchanged. This creates a separate exact E26/E27 evidence layer.
        """
        for role, metrics in (('reviews', self.review_stats),
                              ('metadata', self.meta_stats)):
            for metric, distribution in metrics.items():
                distribution.on_value = (
                    lambda value, role=role, metric=metric:
                    recorder.add(category, role, metric, value))

    @staticmethod
    def summarize_field_states(states, total, expected):
        output = {}
        for field in sorted(set(expected) | set(states)):
            counted = sum(states[field].values())
            if counted > total:
                raise ValueError("E08 field-state counts exceed input population")
            counts = dict(states[field]); counts["absent"] = total - counted
            if sum(counts.values()) != total:
                raise ValueError("E08 field-state conservation violation")
            output[field] = [dict(state=kind, **frequency(counts.get(kind,0),total))
                             for kind in q.MISSING_STATES]
        return output

    @staticmethod
    def _raw_states(row, counter):
        for field, value in row.items():
            counter[field][q.missing_state(value)] += 1

    def review(self, row):
        self.reviews += 1
        self._raw_states(row, self.review_field_states)
        parent = row.get("parent_asin")
        parent = parent if s.indexable_id(parent) else None
        user = row.get("user_id")
        user = user if s.indexable_id(user) else None
        raw_t = row.get("timestamp")
        instant, _ = s.epoch_to_utc(raw_t, self.unit)
        rating = row.get("rating")
        star = int(rating) if q.number(rating) and rating in (1, 2, 3, 4, 5) else None
        title_state = text_state(row, "title")
        body_state = text_state(row, "text")
        title_length = len(row["title"]) if title_state == "nonblank" else None
        body_length = len(row["text"]) if body_state == "nonblank" else None
        body_words = len(row["text"].split()) if body_length is not None else None
        votes = row.get("helpful_vote")
        votes = votes if q.integer(votes) and votes >= 0 else None
        verified = row.get("verified_purchase")
        verified = verified if type(verified) is bool else None
        any_text = title_length is not None or body_length is not None
        image = q.nested_has_usable_content(row.get("images"), "review_image_http_url")
        flags = dict(rating=star is not None, timestamp=instant is not None, title=title_length is not None,
                     body=body_length is not None, any_text=any_text, review_image_url=image,
                     parent=parent is not None, interaction_keys=parent is not None and user is not None
                     and s.indexable_id(row.get("asin")),
                     helpful_vote=votes is not None, verified_purchase=verified is not None)
        for name, flag in flags.items():
            self.review_modalities[name] += flag
        # The primary nonblank-only series are kept for backwards compatibility.
        for name, value in zip(("rating", "title_chars", "body_chars", "body_words", "helpful_vote"),
                               (star, title_length, body_length, body_words, votes)):
            self.review_stats[name].add(value)
        for field, char_key, word_key in (("title", "title_chars_all_strings", "title_words_all_strings"),
                                          ("text", "body_chars_all_strings", "body_words_all_strings")):
            value = row.get(field)
            if isinstance(value, str):
                self.review_stats[char_key].add(len(value))
                self.review_stats[word_key].add(len(value.split()))
        if isinstance(row.get("images"), list):
            self.review_stats["review_image_entries"].add(len(row["images"]))
        self.verified_states["unusable" if verified is None else str(verified).lower()] += 1
        self.text_states["title"][title_state] += 1
        self.text_states["text"][body_state] += 1
        mask = (parent is None) + 2 * (instant is None) + 4 * (star is None)
        self.core_masks[mask] += 1
        if mask == 0:
            self.core_stars[star] += 1
        # E18: script-like characters are not labels of human languages.
        # This includes empty strings in the audit denominator, separately
        # from the legacy nonblank-only body flags below.
        if isinstance(row.get('text'), str):
            from .text_characteristics import body_signature
            signature = body_signature(row['text'])
            self.string_body_records += 1
            self.script_family_records[signature['script_family_pattern']] += 1
            self.script_family_letters.update(signature['family_characters'])
            for name in ('replacement_characters','surrogate_codepoints',
                         'control_codepoints','format_codepoints'):
                self.character_anomalies[name] += signature[name]
                self.character_anomalies[name+'_body_records'] += signature[name] > 0
        if body_length is not None:
            body = row["text"]
            self.body_flags["at_most_3_words"] += body_words <= 3
            self.body_flags["over_10000_codepoints"] += body_length > 10000
            self.body_flags["non_ascii"] += not body.isascii()
            self.body_flags["replacement_character"] += "\ufffd" in body
            # The flag is descriptive and leaves text untouched.
            self.body_flags["surrogate_codepoint"] += bool(SURROGATE.search(body))
        year = None
        if instant is not None:
            year = instant.year
            self.years[year] += 1
            self.months[f"{instant.year:04d}-{instant.month:02d}"] += 1
            self.min_ts = raw_t if self.min_ts is None else min(self.min_ts, raw_t)
            self.max_ts = raw_t if self.max_ts is None else max(self.max_ts, raw_t)
        return parent, user, raw_t if instant is not None else None, star, title_state, body_state, body_length, votes, verified, any_text, year

    def meta(self, row):
        self.metadata += 1
        self._raw_states(row, self.meta_field_states)
        path = row.get("categories")
        if "categories" not in row:
            state = "absent"
        elif path is None:
            state = "null"
        elif not isinstance(path,list):
            state = "wrong_type"
        elif not path:
            state = "empty_list"
        elif all(isinstance(v,str) and v.strip() for v in path):
            state = "valid_full_path"
        else:
            state = "partial_or_invalid_elements"
        self.category_path_states[state] += 1
        price = row.get("price")
        average = row.get("average_rating")
        count = row.get("rating_number")
        price = price if q.number(price) and price >= 0 else None
        average = average if q.number(average) and 1 <= average <= 5 else None
        count = count if q.integer(count) and count >= 0 else None
        image = q.nested_has_usable_content(row.get("images"), "product_image_http_url")
        parent = row.get("parent_asin")
        parent = parent if s.indexable_id(parent) else None
        content = s.metadata_modality_flags(row)[0]
        for name, flag in dict(valid_price=price is not None, average_rating=average is not None,
                               rating_number=count is not None, product_image_url=image,
                               parent=parent is not None, descriptive_content=content).items():
            self.meta_modalities[name] += flag
        for name, value in (("price",price),("average_rating",average),("rating_number",count)):
            self.meta_stats[name].add(value)
        if isinstance(row.get("title"), str):
            self.meta_stats["title_chars_all_strings"].add(len(row["title"]))
        for field in ("features", "description", "images", "videos", "categories", "bought_together"):
            value = row.get(field)
            if isinstance(value, list):
                self.meta_stats[field+"_entries"].add(len(value))
                if field in ("features", "description") and all(isinstance(x,str) for x in value):
                    self.meta_stats[field+"_chars"].add(sum(map(len,value)))
        if isinstance(row.get("details"),dict):
            self.meta_stats["details_keys"].add(len(row["details"]))
        return parent, image, price is not None, content

    def result(self):
        # Months absent within the observed span are explicit zeros, not missing observations.
        months = []
        if self.months:
            from datetime import date
            first, last = min(self.months), max(self.months)
            y, m = map(int, first.split("-"))
            while (y,m) <= tuple(map(int,last.split("-"))):
                name = f"{y:04d}-{m:02d}"
                months.append(dict(month=name, **frequency(self.months[name],sum(self.years.values()))))
                m += 1
                if m == 13: m=1; y+=1
        valid_stars = self.review_stats["rating"]
        valid_time = sum(self.years.values())
        time_min, _ = s.epoch_to_utc(self.min_ts, self.unit)
        time_max, _ = s.epoch_to_utc(self.max_ts, self.unit)
        classes = {str(k):valid_stars.bins[k] for k in range(1,6)}
        population = valid_stars.n
        positive = [v for v in classes.values() if v]
        highest = max(classes.values(),default=0)
        rating_balance = {
            "observation_unit":"parsed review with valid integral rating; raw duplicate copies retained",
            "all_five_classes":{k: frequency(v,population) for k,v in classes.items()},
            "low_1_2":frequency(classes["1"]+classes["2"],population),
            "neutral_3":frequency(classes["3"],population),
            "high_4_5":frequency(classes["4"]+classes["5"],population),
            "missing_classes":[k for k,v in classes.items() if not v],
            "max_to_min_observed_ratio":highest/min(positive) if positive else None,
            "max_to_min_all_five_ratio":highest/min(positive) if len(positive)==5 else None,
            "normalized_entropy_five_classes":(-sum((v/population)*math.log(v/population) for v in positive)/math.log(5)) if population else None,
            "not_future_product_target":True,
        }
        return {
            "reviews": {"unit": "parsed review object; duplicate copies retained",
                        "total": self.reviews,
                        "numeric": {key: dict(stat.summary(self.reviews), measurement_unit=REVIEW_METRICS[key][1])
                                    for key, stat in self.review_stats.items()},
                        "low_star": frequency(valid_stars.bins[1] + valid_stars.bins[2], valid_stars.n),
                        "rating_balance":rating_balance,
                        "text_thresholds_q23":{
                            "short_nonblank_words_at_most_3":frequency(self.body_flags["at_most_3_words"],self.review_stats["body_chars"].n),
                            "long_all_string_codepoints_more_than_10000":frequency(self.body_flags["over_10000_codepoints"],self.review_stats["body_chars_all_strings"].n),
                            "contract":"Q23: short count over nonblank body strings; long count over all string body values; raw review denominator available in reviews.total"},
                        "modalities": {key: frequency(n, self.reviews) for key, n in self.review_modalities.items()},
                        "text_states": {key: state_table(count, self.reviews, TEXT_STATES)
                                        for key, count in self.text_states.items()},
                        "verified_states": state_table(self.verified_states, self.reviews, ("false", "true", "unusable")),
                        "body_flags": {key: frequency(self.body_flags[key], self.review_stats["body_chars"].n)
                                       for key in BODY_FLAGS},
                        "script_character_audit_e18": dict(
                            population='all string review bodies, including empty strings; one record per parsed review object',
                            string_body_records=frequency(self.string_body_records,self.reviews),
                            script_family_by_body={k:frequency(v,self.string_body_records)
                                for k,v in sorted(self.script_family_records.items())},
                            alphabetic_characters_by_family=dict(sorted(self.script_family_letters.items())),
                            anomalies=dict(self.character_anomalies),
                            language_model_status='NOT_RUN_NO_VALIDATED_VERSIONED_MODEL',
                            detected_languages=None,
                            note='Approximate Unicode name-family inspection, not a human-language identifier; Arabic-script text may be Persian or another language. A replacement symbol is an observable character, not proof of when corruption happened.'),
                        "field_states_q01": self.summarize_field_states(self.review_field_states,self.reviews,q.EXPECTED["reviews"]),
                        "core_failure_masks": {str(mask): frequency(self.core_masks[mask], self.reviews)
                                               for mask in range(8)},
                        "core_mask_definition": "1: unusable parent; 2: uninterpretable time; 4: invalid rating; 0: all valid",
                        "valid_core_star_counts_raw": {str(k): self.core_stars[k] for k in range(1, 6)},
                        "years": [dict(year=y, **frequency(n, valid_time))
                                  for y, n in sorted(self.years.items())],
                        "months":months,
                        "month_definition":"UTC calendar months; zero-count interior months included; first/last months may be partial",
                        "valid_time": frequency(valid_time, self.reviews),
                        "time_min_raw": self.min_ts, "time_max_raw": self.max_ts,
                        "time_min_utc": time_min.isoformat() if time_min is not None else None,
                        "time_max_utc": time_max.isoformat() if time_max is not None else None,
                        "timestamp_unit": self.unit},
            "metadata": {"unit": "parsed metadata object; duplicate copies retained",
                         "total": self.metadata,
                         "numeric": {key: dict(stat.summary(self.metadata), measurement_unit=META_METRICS[key][1])
                                     for key, stat in self.meta_stats.items()},
                         "modalities": {key: frequency(n, self.metadata) for key, n in self.meta_modalities.items()},
                         "field_states_q01": self.summarize_field_states(self.meta_field_states,self.metadata,q.EXPECTED["metadata"]),
                         "category_path_states": {name: frequency(self.category_path_states[name],self.metadata)
                                                  for name in ("absent","null","wrong_type","empty_list",
                                                               "partial_or_invalid_elements","valid_full_path")},
                         "category_path_definition": "Exactly one ordered path per metadata row; valid only for a nonempty list of nonblank strings. Levels are 1-based and preserve raw labels.",
                         "role": "snapshot description only"},
        }

