"""Disk-backed EDA populations; never emit IDs or canonical row contents."""
from collections import Counter
import hashlib
import json
import math
import sqlite3
import zlib

import quality_audit as q
from .exact_numeric import ExactNumeric, review_core_metrics, metadata_metrics
import structure_audit as s


SCHEMA = """
CREATE TABLE membership (
    cat TEXT, parent TEXT, raw_n INTEGER, valid_star INTEGER, low INTEGER, core INTEGER,
    PRIMARY KEY(cat,parent)
) WITHOUT ROWID;
CREATE TABLE review (
    digest BLOB, payload BLOB, parent TEXT, usr TEXT, time_key TEXT, star INTEGER,
    has_text INTEGER, year INTEGER, cat TEXT
);
CREATE INDEX review_digest ON review(digest);
CREATE TABLE meta (
    cat TEXT, parent TEXT, digest BLOB, payload BLOB, images INTEGER, price INTEGER,
    content INTEGER, price_band TEXT, copies INTEGER DEFAULT 1
);
CREATE INDEX meta_key ON meta(cat,parent,digest);
CREATE TABLE user_activity (cat TEXT, usr TEXT, n INTEGER NOT NULL,
    PRIMARY KEY (cat,usr)) WITHOUT ROWID;
CREATE TABLE nominal_meta (cat TEXT, field TEXT, label TEXT, n INTEGER NOT NULL,
    PRIMARY KEY (cat,field,label)) WITHOUT ROWID;
"""


def identity(row):
    canonical = q.canonical(row)
    return hashlib.sha256(canonical).digest(), zlib.compress(canonical, level=1)


class Index:
    def __init__(self, path, sequence_enabled):
        self.sequence_enabled = sequence_enabled
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-65536")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript(SCHEMA)
        self.exact_numeric = ExactNumeric(self.db)
        self.pending = {}
        self.review_copies = Counter()
        self.meta_copies = Counter()
        self.user_pending = Counter()
        self.nominal_pending = Counter()

    def flush(self):
        self.exact_numeric.flush()
        self.db.executemany(
            """INSERT INTO membership VALUES (?,?,?,?,?,?)
               ON CONFLICT(cat,parent) DO UPDATE SET raw_n=raw_n+excluded.raw_n,
                   valid_star=valid_star+excluded.valid_star, low=low+excluded.low,
                   core=core+excluded.core""",
            ((cat, parent, *values) for (cat, parent), values in self.pending.items()))
        self.pending.clear()
        self.db.executemany(
            """INSERT INTO user_activity VALUES (?,?,?)
            ON CONFLICT(cat,usr) DO UPDATE SET n=n+excluded.n""",
            ((cat,usr,n) for (cat,usr),n in self.user_pending.items()))
        self.user_pending.clear()
        self.db.executemany(
            """INSERT INTO nominal_meta VALUES (?,?,?,?)
            ON CONFLICT(cat,field,label) DO UPDATE SET n=n+excluded.n""",
            ((cat,field,label,n) for (cat,field,label),n in self.nominal_pending.items()))
        self.nominal_pending.clear()
        self.db.commit()

    def review(self, cat, row, parent, user, stamp, star, has_text, year):
        if user is not None:
            self.user_pending[(cat,user)] += 1  # Includes valid-user reviews with invalid parent.
        if parent is None:
            return
        values = self.pending.setdefault((cat, parent), [0, 0, 0, 0])
        values[0] += 1
        values[1] += star is not None
        values[2] += star in (1, 2)
        values[3] += stamp is not None and star is not None
        # E41/E44: Only sequence-eligible (valid timestamp AND valid star)
        # records need canonical identity and conflict indexing. Invalid-star
        # records remain counted in the raw audit but cannot create a conflict
        # that excludes an otherwise eligible product.
        if not self.sequence_enabled or stamp is None or star is None:
            return
        # E09: global review-core raw versus globally unique parsed records.
        # This population is NOT all raw review objects; keep its label explicit.
        self.exact_numeric.add('GLOBAL', 'core_raw', 'rating', star)
        for metric, value in review_core_metrics(row):
            self.exact_numeric.add('GLOBAL', 'core_raw', metric, value)
        digest, payload = identity(row)
        for (other,) in self.db.execute("SELECT payload FROM review WHERE digest=?", (digest,)):
            # Byte equality is verified even if different records share SHA-256.
            if other == payload:
                if star is not None:
                    self.review_copies[cat] += 1
                return
        self.db.execute("INSERT INTO review VALUES (?,?,?,?,?,?,?,?,?)",
                        (digest, payload, parent, user, s.sortable_timestamp_key(stamp), star,
                         has_text, year, cat))
        self.exact_numeric.add('GLOBAL', 'core_unique', 'rating', star)
        for metric, value in review_core_metrics(row):
            self.exact_numeric.add('GLOBAL', 'core_unique', metric, value)

    def meta(self, cat, row, parent, images, price, content):
        for field in ("main_category","store"):
            label=row.get(field)
            if isinstance(label,str) and label.strip():
                # Preserve exact Unicode labels; never trim or merge raw labels.
                self.nominal_pending[(cat,field,label)] += 1
        # E23/E28: Amazon's `categories` is an ordered path, not an
        # unordered bag of independent category labels. Each valid path
        # contributes once per position and once as a full-path category.
        # Partial paths do NOT silently become complete paths.
        hierarchy = row.get("categories")
        if isinstance(hierarchy, list):
            valid_path = bool(hierarchy)
            for depth, label in enumerate(hierarchy, start=1):
                if isinstance(label, str) and label.strip():
                    self.nominal_pending[(cat, f"categories.level_{depth}", label)] += 1
                else:
                    valid_path = False
            if valid_path:
                label = json.dumps(hierarchy, ensure_ascii=True, separators=(",", ":"))
                self.nominal_pending[(cat, "categories.full_path", label)] += 1
        if parent is None:
            return
        # E09: exact parsed-record deduplication inside each category/parent,
        # restricted to metadata with a valid parent identifier.
        for metric, value in metadata_metrics(row):
            self.exact_numeric.add(cat, 'parent_metadata_raw', metric, value)
        digest, payload = identity(row)
        for row_id, other in self.db.execute(
            "SELECT rowid,payload FROM meta WHERE cat=? AND parent=? AND digest=?", (cat, parent, digest)
        ):
            if payload == other:
                self.db.execute("UPDATE meta SET copies=copies+1 WHERE rowid=?", (row_id,))
                self.meta_copies[cat] += 1
                return
        # E38: fixed, descriptive price ranges in the raw units supplied in
        # metadata. No currency, inflation adjustment or price-at-t5 is implied.
        value = row.get('price')
        if not q.number(value) or value < 0:
            price_band = None
        elif value == 0:
            price_band = '0'
        elif value <= 25:
            price_band = '(0,25]'
        elif value <= 100:
            price_band = '(25,100]'
        elif value <= 500:
            price_band = '(100,500]'
        else:
            price_band = '(500,+inf)'
        self.db.execute("INSERT INTO meta(cat,parent,digest,payload,images,price,content,price_band) VALUES (?,?,?,?,?,?,?,?)",
                        (cat, parent, digest, payload, images, price, content, price_band))
        for metric, value in metadata_metrics(row):
            self.exact_numeric.add(cat, 'parent_metadata_unique', metric, value)

    def prepare(self):
        self.db.execute("CREATE INDEX membership_by_parent ON membership(parent,cat)")
        self.db.executescript("""
            CREATE TABLE meta_parent (
                cat TEXT, parent TEXT, versions INTEGER, copies INTEGER,
                image INTEGER, price INTEGER, content INTEGER, price_band TEXT, PRIMARY KEY(cat,parent)
            ) WITHOUT ROWID;
            INSERT INTO meta_parent
                SELECT cat,parent,COUNT(*),SUM(copies),MAX(images),MAX(price),MAX(content),MAX(price_band)
                FROM meta GROUP BY cat,parent;
        """)
        if self.sequence_enabled:
            self.db.executescript("""
                CREATE INDEX candidate_event ON review(parent,usr,time_key) WHERE usr IS NOT NULL;
                CREATE TABLE conflicts (parent TEXT PRIMARY KEY, groups_n INTEGER, versions_n INTEGER) WITHOUT ROWID;
                INSERT INTO conflicts
                    SELECT parent,COUNT(*),SUM(versions) FROM (
                        SELECT parent,usr,time_key,COUNT(*) AS versions FROM review
                        WHERE usr IS NOT NULL AND star IS NOT NULL
                        GROUP BY parent,usr,time_key
                        HAVING COUNT(*) > 1
                    ) GROUP BY parent;
                DROP INDEX candidate_event;
                CREATE INDEX sequence_order ON review(parent,time_key,digest) WHERE star IS NOT NULL;
            """)
        self.db.commit()

    def raw_products(self, cat):
        return self.db.execute("""
            SELECT m.raw_n,m.valid_star,m.low,m.core,p.versions,p.copies,p.image,p.price,p.content,p.price_band
            FROM membership m LEFT JOIN meta_parent p ON p.cat=m.cat AND p.parent=m.parent
            WHERE m.cat=?
        """, (cat,))

    def metadata_keys(self, cat):
        return self.db.execute("""
            SELECT COUNT(*),COUNT(CASE WHEN m.parent IS NULL THEN 1 END),
                   COUNT(CASE WHEN p.versions>1 THEN 1 END)
            FROM meta_parent p LEFT JOIN membership m ON p.cat=m.cat AND p.parent=m.parent
            WHERE p.cat=?
        """, (cat,)).fetchone()

    def memberships(self):
        return self.db.execute("SELECT parent,cat,core FROM membership ORDER BY parent,cat")

    def reviews(self):
        return self.db.execute("""
            SELECT parent,time_key,star,has_text,year,usr FROM review WHERE star IS NOT NULL
            ORDER BY parent,time_key,digest,rowid
        """)

    def exact_numeric_profiles(self, cat, raw):
        """Full raw metric summaries; status 'not exact' is explicit per metric."""
        result={}
        for role in ('reviews','metadata'):
            result[role]={}
            for metric, summary in raw[role]['numeric'].items():
                result[role][metric]=self.exact_numeric.profile(
                    cat,role,metric,summary['valid'])
        return result

    def global_core_numeric_sensitivity(self):
        """Cross-category exact canonical duplicate view for valid sequence core."""
        if not self.sequence_enabled:
            return {'status':'not_available_no_sequence_index'}
        return dict(status='computed',scope='global valid parent/time/rating sequence core',
            metrics=self._compare_roles(
                'GLOBAL','core_raw','core_unique',
                ('rating','helpful_vote','body_chars_all_strings',
                 'body_words_all_strings','title_chars_all_strings','review_image_entries')))

    def metadata_numeric_sensitivity(self, cat):
        return dict(status='computed',
            scope='one category metadata with indexable parent; exact canonical dedup within category and parent',
            metrics=self._compare_roles(cat,'parent_metadata_raw','parent_metadata_unique',
                       ('price','average_rating','rating_number','images_entries')))

    def _compare_roles(self, category, before, after, metrics):
        output={}
        for metric in metrics:
            nr=self.exact_numeric.count(category,before,metric)
            nu=self.exact_numeric.count(category,after,metric)
            if nu>nr:
                raise ValueError('E09 deduplicated metric exceeds raw observations')
            lhs=self.exact_numeric.profile(category,before,metric,nr)
            rhs=self.exact_numeric.profile(category,after,metric,nu)
            tv=self.exact_numeric.total_variation(category,before,after,metric,nr,nu)
            output[metric]=dict(raw=lhs,exact_unique=rhs,
                removed_valid_observations=nr-nu, total_variation_distance=tv,
                interpretation='Different denominator after canonical dedup; not a causal change or all-input sensitivity')
        return output

    def unique_core_star_counts(self):
        """Five compact counts from disk, after global exact parsed-row deduplication."""
        if not self.sequence_enabled:
            return None
        counts = Counter()
        for star, n in self.db.execute(
                "SELECT star, COUNT(*) FROM review WHERE star IS NOT NULL GROUP BY star"):
            counts[star] = n
        return {str(star): counts[star] for star in range(1, 6)}

    def conflicts(self):
        return self.db.execute("SELECT parent,groups_n,versions_n FROM conflicts ORDER BY parent")

    def nominal_fields(self, cat):
        """All observed category-path depths; keep core nominal fields even when empty."""
        fields={"main_category", "store", "categories.full_path"}
        fields.update(field for (field,) in self.db.execute(
            "SELECT DISTINCT field FROM nominal_meta WHERE cat=?", (cat,)))
        def order(field):
            if field.startswith('categories.level_'):
                suffix=field.removeprefix('categories.level_')
                return (2, int(suffix) if suffix.isdecimal() else 10**9)
            return ({'main_category':0, 'store':1, 'categories.full_path':3}.get(field,4),0)
        return sorted(fields, key=order)

    def nominal_profile(self, cat, field, total, top_k=12):
        """Exact on-disk frequencies AND Shannon/Simpson imbalance over ALL levels."""
        if field not in self.nominal_fields(cat):
            raise ValueError("unknown nominal field")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        distinct, valid = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(n),0) FROM nominal_meta WHERE cat=? AND field=?",
            (cat,field)).fetchone()
        rows = self.db.execute(
            "SELECT label,n FROM nominal_meta WHERE cat=? AND field=? ORDER BY n DESC,label LIMIT ?",
            (cat,field,top_k)).fetchall()
        # Iterate the database cursor without materialising the high-cardinality
        # label vocabulary. Entropy and Simpson are defined over ALL levels,
        # not just the first `top_k` labels shown in the graph.
        entropy_sum = 0.0
        simpson_sum = 0.0
        if valid:
            for (count,) in self.db.execute(
                "SELECT n FROM nominal_meta WHERE cat=? AND field=?", (cat,field)):
                share = count / valid
                entropy_sum -= share * math.log(share)
                simpson_sum += share * share
        from .stats import frequency
        return dict(unit=("one metadata record per exact ordered category path" if field=="categories.full_path"
                          else "one metadata record at the specified level (exact, unnormalized label)" if field.startswith("categories.level_")
                          else "raw metadata record; exact Unicode label; no normalization"),
                    valid=valid,excluded=total-valid,distinct_levels=distinct,
                    shannon_entropy_nats=entropy_sum if valid else None,
                    normalized_shannon_entropy=(entropy_sum / math.log(distinct) if distinct>1 else 0.0 if distinct==1 else None),
                    simpson_concentration=simpson_sum if valid else None,
                    effective_level_count=math.exp(entropy_sum) if valid else None,
                    imbalance_scope="all observed exact labels, not just plotted top levels",
                    mode=rows[0][0] if rows else None,
                    top=[dict(label=label,**frequency(n,valid)) for label,n in rows],
                    other=dict(levels=max(0,distinct-len(rows)),**frequency(valid-sum(n for _,n in rows),valid)),
                    missing_or_unusable=frequency(total-valid,total))

    def user_volume_profile(self, cat, total_review_rows):
        """Reviewer counts per category, disk-backed and raw-copy weighted."""
        from .stats import Distribution, frequency
        from .stats import volume_bin
        import math
        d=Distribution()
        classes=Counter()
        group_total=0; source_rows=0
        for value, groups in self.db.execute(
                "SELECT n,COUNT(*) FROM user_activity WHERE cat=? GROUP BY n ORDER BY n",(cat,)):
            d.add_many(value,groups)
            classes[volume_bin(value)] += groups
            group_total += groups
            source_rows += value*groups
        top_k=math.ceil(group_total*.01)
        remaining=top_k; top_sum=0
        if remaining:
            for value, groups in self.db.execute(
                    "SELECT n,COUNT(*) FROM user_activity WHERE cat=? GROUP BY n ORDER BY n DESC",(cat,)):
                take=min(remaining,groups)
                top_sum+=take*value
                remaining-=take
                if remaining==0: break
        return dict(unit="one exact reviewer ID per category; raw review objects retained",
                    reviewer_count=group_total,
                    source_review_rows_with_valid_user=frequency(source_rows,total_review_rows),
                    activity_distribution=d.summary(group_total),
                    volume_groups={key:frequency(classes[key],group_total)
                                   for key in ("1–4","5–14","15–49","50–199","200+")},
                    top_one_percent_reviewer_count=top_k,
                    top_one_percent_raw_review_share=frequency(top_sum,source_rows))
