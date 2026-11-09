"""CLI for a single four-category EDA run."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import time
from uuid import uuid4

import quality_audit as quality
import structure_audit as structure

from . import VERSION
from .index import Index
from .input_data import load_config, scan
from .input_roles import validate_dataset_roles
from .artifacts import write_inventory
from .quality_link import reconcile_quality
from .multivariate import ReviewRelations
from .products import POLICIES, raw_product_results, reconcile
from .precision_report import write_precision
from .report import category_reports, distribution_csv, markdown, percent, rate_csv, write_csv, write_json
from .research_contract import load_research_contract, research_markdown, CONTRACT_FILENAME
from .stats import frequency
from .univariate import RawCategory
from .split_design import document as split_document


GIB = 1024 ** 3


def source_manifest():
    project = Path(__file__).resolve().parent.parent
    files = [*sorted((project / "eda").glob("*.py")), Path(quality.__file__),
             Path(structure.__file__), project / "PROJECT_CARD.md", project / CONTRACT_FILENAME]
    return {str(path.relative_to(project)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def estimate(datasets, root, mode, limit, expected_records, min_gib):
    stored = sum(path.stat().st_size for paths in datasets.values() for path in paths.values())
    count_overhead = expected_records * 640 if expected_records is not None else None
    # Conservative planning heuristic, NOT an upper bound. E26/E27 add a
    # disk-backed (category, role, metric, value) frequency B-tree.
    rough = stored * 4 + count_overhead if count_overhead is not None else None
    return dict(mode=mode, physical_line_limit_per_file=limit or None, stored_input_bytes=stored,
                expected_input_records_user_supplied=expected_records,
                heuristic_scratch_bytes=rough, formula="4 * stored input bytes + 640 * expected total records; planning heuristic, NOT a bound",
                enforced_free_space_floor_bytes=int(min_gib * GIB), initial_free_bytes=shutil.disk_usage(root).free,
                warning="Planning heuristic only. Exact numeric frequency B-tree adds disk writes and sorting work. Cannot bound compression, cardinality, key width, SQLite temporary space or runtime. Pilot a representative subset, monitor free space, and reserve additional headroom.",
                complexity="one sequential raw parse per input; SQLite B-tree insert/sort typically O(R log R + P log P); sequential product/sequence aggregation; no global row list")


def quality_evidence(directory, inputs, scope, unit):
    if directory is None:
        return dict(status="not_provided", reused_metrics=[])
    source = Path(directory).resolve()
    result = {}
    for cat, files in inputs.items():
        path = source / cat / "quality_summary.json"
        if not path.is_file():
            result[cat] = {"status": "missing"}
            continue
        try:
            payload = path.read_bytes()
            item = json.loads(payload)
            evidence_unit = item["structure_audit_integration"]["timestamp_unit"]
            reasons = []
            if item.get("audit_version") != quality.VERSION:
                reasons.append("quality_version")
            if item["structure_audit_integration"].get("module_version") != structure.VERSION:
                reasons.append("structure_version")
            if item.get("source_category") != cat:
                reasons.append("category")
            if item.get("scope") != scope or scope != "entire_supplied_dataset_files":
                reasons.append("scope")
            if evidence_unit != unit:
                reasons.append("timestamp_unit")
            for role, info in files.items():
                reported = item["files"][role]
                if reported["scanned_content_sha256"] != info["scanned_content_sha256"]:
                    reasons.append(role+"_content_sha256")
                if reported["object_records"] != info["counts"]["objects"]:
                    reasons.append(role+"_object_count")
            result[cat] = dict(status="unverified" if reasons else "matched_input_and_scope",
                               mismatch=reasons, evidence_sha256=hashlib.sha256(payload).hexdigest())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            result[cat] = {"status": "unreadable_or_incompatible"}
    return dict(status="comparison_only", by_category=result, reused_metrics=[],
                note="Fingerprint match does not certify a quality decision or definition equivalence.")


def export(folder, inputs, products, raw, relations, cohorts, global_summary, scope, policy):
    # Imported here so preflight and JSON/SQLite setup are independent of the plotting backend.
    from . import plots

    rate_csv(folder / "global_product_rates.csv", global_summary)
    write_json(folder / "global_products.json", global_summary)
    # E51 is an explicit design-only deliverable. No dates are guessed from the
    # target and no train/test assignment or model training occurs in Phase 1.
    split=split_document()
    write_json(folder / "SPLIT_DESIGN_E51.json",split)
    markdown(folder / "SPLIT_DESIGN_E51_FA.md", [
        "# طرح تقسیم زمانی بدون نشت (E51)", "",
        "**وضعیت: فقط طراحی؛ تاریخ‌های برش هنوز انتخاب نشده و هیچ تقسیم واقعی اجرا نشده است.**", "",
        "واحد تقسیم، محصول جهانی یکتا با شناسهٔ `parent_asin` است. ابتدا کپی‌های دقیق بین چهار فایل حذف و تعارض‌ها و تساوی مرزی تعیین می‌شوند.", "",
        "برای هر محصول، بلوک زمانی صرفاً با زمان نقد پنجم (`t5`) تعیین می‌شود. ورود برچسب به هر بلوک فقط وقتی مجاز است که نقد پانزدهم (`t15`) تا پایان همان بلوک مشاهده شده باشد.", "",
        "محصول بدون ۱۵ نقد معتبر یا با برچسب نارسیده **صفر برچسب نمی‌گیرد** و به بلوک زمانی بعدی منتقل نمی‌شود.", "",
        "## شروط اجباری", "", *[f"- {r}" for r in split['mandatory_rules']], "",
        "## کارهای باز", "", *[f"- {r}" for r in split['pending']], "",
        "این مستند، ارزیابی S در برابر ST یا آزمون نهایی را انجام‌شده معرفی نمی‌کند."
    ])
    # E50: a fully labelled change-in-population comparison. A policy
    # difference does NOT estimate the causal effect of conflicting events.
    policy_rows={}
    if 'exclude' in cohorts and 'retain' in cohorts:
        for cat in sorted(cohorts['exclude']):
            a,b=cohorts['exclude'][cat],cohorts['retain'][cat]
            x,y=(a['mutually_exclusive_outcomes']['eligible']['numerator'],
                 b['mutually_exclusive_outcomes']['eligible']['numerator'])
            if y<x:
                raise AssertionError('Retaining candidates cannot reduce eligible population')
            ax=a['future_low_share'].get('mean')
            by=b['future_low_share'].get('mean')
            policy_rows[cat]=dict(
                primary_category=cat, exclude_eligible_products=x,
                retain_eligible_products=y, newly_included_when_retaining=y-x,
                candidate_event_groups=a['candidate_event_groups'],
                ties_excluded_in_both=sum(a['mutually_exclusive_outcomes'][key]['numerator']
                                          for key in ('tie_at_5_only','tie_at_15_only','both_boundary_ties')),
                exclude_mean_future_low_share=ax,
                retain_mean_future_low_share=by,
                unpaired_population_mean_difference=(by-ax) if ax is not None and by is not None else None,
                exclusive_category_sensitivity=a['no_overlap_sensitivity'],
                first15_candidate_conflict_sensitivity=a['window15_conflict_policy_sensitivity'],
                t5_year_strata={year:dict(
                    mature_eligible_products=dist['valid'],
                    observed_future_low_share=dist['mean'],
                    nonmature_or_excluded_products_not_assigned_zero=True)
                    for year,dist in a['target_by_t5_year'].items()},
                interpretation='Unpaired cohort selection and calendar-stratum descriptions; not paired prediction improvement, causal impact, or a justified main-policy replacement')
    write_json(folder / 'cohort_policy_comparison.json', dict(
        scope=scope, status='computed' if policy_rows else 'not_computed_outside_full_or_synthetic',
        unit='one globally unique product assigned to one primary category',
        policy_comparison=policy_rows, earliest_event='t5 fifth valid observed review',
        notes=['Different policy populations mean their overall target averages are NOT a paired effect.',
               'Exact-copy sensitivity is separate and recorded under global_products.json.',
               'Boundary ties are excluded in BOTH policies; no arbitrary chronology is assumed.',
               'Excluding multi-category products changes the population, not the outcomes of retained products.',
               'Observed t5-year strata are descriptive, CONDITIONAL ON 15 reviews and other eligibility; not a time-cutoff or held-out performance sensitivity.',
               'Time-block sensitivity with calendar cutoffs still awaits predeclared dates and real follow-up coverage.']))
    lines=['# حساسیت جامعهٔ محصولی ۵+۱۰ به سیاست‌ها (E50)', '',
           f'دامنه: `{scope}`. این جدول‌ها فقط در اجرای دارای توالی معتبر محاسبه می‌شوند.', '',
           '| دستهٔ اصلی | واجد شرایط: حذف تعارض | واجد شرایط: نگهداری | اختلاف تعداد | میانگین هدف با حذف | میانگین هدف با نگهداری | تغییر میانگین (غیرجفتی) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for cat,v in policy_rows.items():
        fmt=lambda x: 'تعریف‌نشده' if x is None else f'{x:.4f}'
        lines.append(f"| `{cat}` | {v['exclude_eligible_products']} | {v['retain_eligible_products']} | {v['newly_included_when_retaining']} | {fmt(v['exclude_mean_future_low_share'])} | {fmt(v['retain_mean_future_low_share'])} | {fmt(v['unpaired_population_mean_difference'])} |")
    lines += ['', 'تفاوت میانگین‌های دو سیاست **جفت‌شده نیست** زیرا جامعه تغییر می‌کند. جدول جداگانهٔ بدون هم‌پوشانی محصولات در هر `cohort.json` قرار دارد. هیچ تعارضی با انتخاب امتیاز ترجیحی حل نشده است.',
              '', '### سیاست جایگزین تعارض صرفاً در پنجرهٔ ۱۵ نقد',
              'این سیاست فقط یک سناریوی حساسیت است؛ سیاست اصلی حذف تمام تعارض‌های نامزد در توالی مشاهده‌شده بدون تغییر می‌ماند.',
              '| دسته | گروه‌های تعارض در ۱۵ نقد نخست | گروه‌های فاقد دو نسخه در پنجرهٔ ۱۵ نقد | واجدشرایط با قاعدهٔ جایگزین | محصول تازه نسبت به حذف سراسری |',
              '|---|---:|---:|---:|---:|']
    for cat,v in policy_rows.items():
        z=v['first15_candidate_conflict_sensitivity']
        lines.append(f"| `{cat}` | {z['candidate_conflicts_in_first15']} | {z['candidate_groups_not_repeated_in_first15']} | {z['eligible_products_if_only_first15_conflicts_excluded']} | {z['newly_included_vs_global_exclude']} |")
    lines += ['', '### لایه‌بندی تقویمی توصیفی',
              'میانگین‌های درون‌سال **تنها برای محصولاتی که هدف آن‌ها رسیده** محاسبه می‌شوند؛ از آن‌ها نمی‌توان عملکرد تقسیم زمانی یا اثر تقویمی نتیجه گرفت. برش‌های آموزش/اعتبارسنجی/آزمون انتخاب نشده‌اند.',
              '| دسته | سال رسیدن نقد پنجم | واجدشرایط با هدف بالغ | میانگین سهم کم‌ستارهٔ بعدی |',
              '|---|---|---:|---:|']
    for cat,v in policy_rows.items():
        for year,z in sorted(v['t5_year_strata'].items()):
            lines.append(f"| `{cat}` | {year} | {z['mature_eligible_products']} | {z['observed_future_low_share']:.4f} |")
    lines += ['', 'تحلیل حساسیت واقعیِ تغییر تاریخ برش همچنان **انجام نشده** زیرا هیچ تاریخ تقسیم مصوبی وجود ندارد.']
    markdown(folder/'COHORT_POLICY_COMPARISON_FA.md',lines)
    comparisons = {}
    for cat, rv in raw.items():
        part = folder / cat
        part.mkdir()
        meta, reviews = rv["metadata"], rv["reviews"]
        rel, prod = relations[cat], products[cat]
        uni = dict(scope=scope, category=cat, **rv,
                   products={key: val for key, val in prod.items() if key not in ("volume_vs_rating", "metadata_conditioned_ratings")})
        multi = dict(scope=scope, category=cat, review_relations=rel,
                     product_volume_vs_rating=prod["volume_vs_rating"],
                     metadata_connection=prod["metadata_connection"],
                     metadata_conditioned_ratings=prod["metadata_conditioned_ratings"])
        cases = {name: grouped[cat] for name, grouped in cohorts.items()}
        for name, data in (("univariate", uni), ("multivariate", multi)):
            write_json(part / f"{name}.json", data)
            rate_csv(part / f"{name}_rates.csv", data)
        numerics = {**{f"reviews.{k}": d for k, d in reviews["numeric"].items()},
                    "reviewers.activity": dict(reviews["reviewer_activity"]["activity_distribution"],
                                              measurement_unit="raw review objects per valid reviewer ID"),
                    **{f"metadata.{k}": d for k, d in meta["numeric"].items()},
                    **{f"products.{k}": dict(prod[k], measurement_unit="review records per product")
                       for k in ("raw_review_volume", "valid_core_review_volume")},
                    **{f"star_{r['star']}.body_length": dict(r["body_length"], measurement_unit="Unicode code points")
                       for r in rel["by_star"]},
                    **{f"star_{r['star']}.helpful_vote": dict(r["helpful_vote"], measurement_unit="votes")
                       for r in rel["by_star"]}}
        if cases:
            payload = dict(scope=scope, status="provisional", primary_policy=policy, policies=cases)
            write_json(part / "cohort.json", payload)
            rate_csv(part / "cohort_rates.csv", payload)
            for key, row in cases.items():
                numerics[f"{key}.future_low_share"] = dict(row["future_low_share"], measurement_unit="fraction")
                numerics[f"{key}.maturity_days"] = dict(row["maturity_days"], measurement_unit="days")
            plots.cohort_plot(part, cat, cases, policy, scope)
        distribution_csv(part / "distributions.csv", numerics)
        category_reports(part, cat, scope, rv, rel, prod, cases, policy)
        from .univariate_advanced import export_extended
        export_extended(part,cat,raw=rv,scope=scope)
        plots.raw_plots(part, cat, rv, rel, prod, scope)
        plots.multivariate_extended_plot(part, cat, rel, prod, scope)
        plots.missingness_by_subgroup_plot(part, cat, rel, scope)
        comparisons[cat] = {
            "any_text": reviews["modalities"]["any_text"],
            "review_image": reviews["modalities"]["review_image_url"],
            "valid_rating": reviews["modalities"]["rating"],
            "valid_time": reviews["modalities"]["timestamp"],
            "meta_price": meta["modalities"]["valid_price"],
            "meta_image": meta["modalities"]["product_image_url"],
            "low_star": reviews["low_star"], "short_volume": prod["raw_volume_groups"]["1–4"],
            "eligible": cases[policy]["mutually_exclusive_outcomes"]["eligible"] if cases else frequency(0, 0),
        }
    # E38: an auditable, descriptive snapshot price association using only
    # uniquely joined metadata; this is NEVER a historical predictor.
    write_csv(folder/'E38_PRICE_BAND_ASSOCIATIONS.csv',
        ('category','price_band_raw_units','uniquely_matched_products','raw_review_objects_of_those_products',
         'products_with_valid_rating','equal_product_mean_low_star','review_weighted_low_star'),
        ((cat,band['price_band'],band['matched_products'],band['raw_reviews_of_matched_products'],
          band['products_with_valid_ratings']['numerator'],
          band['low_star_equal_product_mean'],band['low_star_review_weighted']['fraction'])
         for cat,prod in sorted(products.items())
         for band in prod['metadata_conditioned_ratings']['price_band_analysis']['bands']))
    write_precision(folder, raw, global_summary)
    write_json(folder / "category_comparison.json", dict(scope=scope, categories=comparisons))
    rate_csv(folder / "category_comparison.csv", comparisons)
    write_csv(folder / "scan_counts.csv", ("category", "role", "measure", "count", "unit"),
              ((c, r, measure, n, "physical lines (parsed objects count once)")
               for c, roles in inputs.items() for r, item in roles.items() for measure, n in item["counts"].items()))
    comparisons_by_star = {cat: [
        dict(star=r["star"], nonblank_body=r["body"][-1],
             nonblank_title=r["title"][-1], zero_helpful_votes=r["helpful_vote"]["zero"],
             verified_purchase=r["verified"][1])
        for r in relation["by_star"]] for cat, relation in relations.items()}
    write_json(folder / "category_by_star.json", dict(scope=scope, categories=comparisons_by_star,
               unit="parsed review object conditional on valid rating in file; zero votes also needs valid votes"))
    rate_csv(folder / "category_by_star.csv", comparisons_by_star)
    plots.comparison_plot(folder, comparisons, comparisons_by_star, relations, scope)
    lines = ["# مقایسهٔ توصیفی چهار دسته", "", f"دامنه `{scope}`", "",
             f"محصول جهانی یکتا در فایل‌های نقد {global_summary['total_unique_products']:,} و "
             f"تعداد عضویت محصول در دسته {global_summary['total_category_product_memberships']:,}. "
             f"محصول دارای عضویت چندگانه {percent(global_summary['multiple_categories'])}.", "",
             "| دسته | متن نقد از اشیای نقد | تصویر نقد از اشیای نقد | قیمت معتبر از اشیای فراداده | "
             "تصویر محصول از اشیای فراداده | امتیاز پایین از نقدهای با امتیاز معتبر | ورود موقت از محصولات دستهٔ اصلی |",
             "|---|---|---|---|---|---|---|"]
    for cat, vals in comparisons.items():
        keys = ("any_text", "review_image", "meta_price", "meta_image", "low_star", "eligible")
        lines.append(f"| `{cat}` | " + " | ".join(percent(vals[k]) for k in keys) + " |")
    lines += ["", "تفاوت مشاهده‌شده در دسترس‌پذیری توصیفی است. از جدول نمی‌توان تصادفی‌بودن گمشدگی، "
              "ترکیب جامعهٔ همهٔ خریداران یا عملکرد مدل را نتیجه گرفت. برای جمعیت، حذف‌ها و نمودارهای هر تحلیل، "
              "گزارش‌های تفکیکی دسته و دفتر تعاریف را کنار جدول بخوانید. مقایسهٔ شرطیِ بدنه در سطح هر ستاره و ترکیب سالانه در نمودار مشترک آمده‌اند."]
    markdown(folder / "CATEGORY_COMPARISON_FA.md", lines)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "DEFINITIONS_FA.md", folder / "DEFINITIONS_FA.md")
    # E56: each observation is calculated from the current run, not from a
    # previous Q archive or a hard-coded generalization.
    interpretations = [
        '# ثبت شواهد، تفسیر و محدودیت‌های EDA (E56)', '',
        f'**دامنهٔ این اجرا:** `{scope}`. اعداد این گزارش فقط از همین اجرا هستند.', '',
        'این تحلیل مشاهده‌ای است؛ اختلاف درصدها دلیل علیت، فروش بیشتر یا عملکرد پیش‌بینی نیست.', '',
        '## پرسش، شاهد و بررسی بعدی برای هر دسته', ''
    ]
    for cat in sorted(raw):
        rv = raw[cat]['reviews']
        md = raw[cat]['metadata']
        low, body, price = (rv['low_star'],rv['modalities']['any_text'],
                            md['modalities']['valid_price'])
        def evidence(label, obj):
            value = ('تعریف‌نشده' if obj['fraction'] is None else
                     f"{100*obj['fraction']:.2f}٪")
            return f"{label}: {obj['numerator']:,} از {obj['denominator']:,} ({value})"
        interpretations += [
            f'### {cat}', '',
            '**پرسش:** توزیع نقدهای کم‌ستاره، وجود متن و دسترس‌پذیری قیمت در دادهٔ مشاهده‌شده چیست؟', '',
            '**شاهد عددی:** '+ '; '.join([
                evidence('امتیاز ۱ یا ۲ در میان امتیازهای معتبر',low),
                evidence('عنوان یا متن ناتهی در کل نقدها',body),
                evidence('قیمت معتبر در کل ردیف‌های فراداده',price)]) + '.', '',
            '**تفسیر مجاز:** این نسبت‌ها تنها سهم رکوردها در جامعه‌های متفاوت را توصیف می‌کنند. قیمتِ فراداده در زمان گردآوری ثبت شده و تاریخی نیست.', '',
            '**توضیح رقیب / محدودیت:** خودگزینی نویسندگان نقد، کپی دقیق، فقدان متن و تفاوت پوشش دسته‌ها می‌توانند الگوهای مشاهده‌شده را تغییر دهند. محصول و نقد نمونه‌های مستقل نیستند.', '',
            '**بررسی بعدی:** نرخ‌ها را روی جامعهٔ محصول واجدشرایط ۵+۱۰، پس از قواعد تکراری/تعارض و با برش زمانی مشخص دوباره بررسی کنید؛ سنجش ارزش افزودهٔ متن به فاز مدل‌سازی تعلق دارد.', ''
        ]
    interpretations += [
        '## کنترل بازبینی نمودارها', '',
        'هر نمودار باید به شاخص مشخص، واحد مشاهده، صورت و مخرج درصد و دامنهٔ اجرا متصل باشد. ',
        'در نمودارهای دارای نمونهٔ نمایشی، حجم نمونه جایگزین حجم آمار کل نیست.', '',
        '**این گزارش جایگزین بازبینی انسانی همهٔ نمودارهای واقعی نیست.**'
    ]
    markdown(folder/'INTERPRETATION_E56_FA.md',interpretations)
    markdown(folder/'PHASE_BOUNDARY_E57_FA.md',[
        '# مرزبندی فاز اول با مراحل بعد (E57)', '',
        f'دامنهٔ این اجرا: `{scope}`.', '',
        '**انجام شده در فاز اول:** توصیف توزیع‌های نقد و فراداده، روابط مشاهده‌ای، جامعهٔ موقت ۵+۱۰، ثبت فرضیه و طراحی اولیهٔ تقسیم.', '',
        '**هنوز انجام نشده:** تقسیم واقعی آموزش/اعتبارسنجی/آزمون با تاریخ برش مصوب، یادگیری واژگان/TF-IDF، استخراج embedding، آموزش خط مبنای S یا مدل ST، تنظیم ابرپارامتر و سنجش MAE/RMSE یا عدم قطعیت اختلاف عملکرد.', '',
        '**اصل جلوگیری از نشت:** هر آمار یادگرفته‌شده از داده (واژگان، نرمال‌سازی، جایگزینی و انتخاب ویژگی) فقط روی بخش آموزش برازش شود. قیمت، میانگین امتیاز و شمار امتیاز فرادادهٔ تجمیعی در زمان پیش‌بینی پنجم به‌طور پیش‌فرض مجاز نیستند.', '',
        '**تفسیر صحیح:** هیچ جدول و نمودار فاز اول به‌تنهایی نشان‌دهندهٔ برتری ST بر S نیست.', '',
        'تعریف محصول جهانی، حذف کپی دقیق و سیاست برخورد با تساوی/تعارض نیز موقت‌اند و در دفتر ممیزی موارد باز ثبت می‌شوند.'
    ])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Amazon Reviews’23 | Phase 1 EDA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=("preview", "full", "synthetic", "plan"))
    parser.add_argument("--preview-lines", type=int)
    parser.add_argument("--out", type=Path, required=True, help="New run directory; existing directories are rejected.")
    parser.add_argument("--conflict-policy", default="exclude", choices=POLICIES)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--sample-size", type=int, default=0, help="Optional display sample per primary category, maximum 10000.")
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--min-free-gib", type=float)
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument("--quality-dir", type=Path, help="Optional directory containing CATEGORY/quality_summary.json.")
    quality_group.add_argument("--quality-archive", type=Path, help="Optional Q01–Q25 output ZIP; read in place without extraction.")
    args = parser.parse_args(argv)
    if args.preview_lines is None:
        args.preview_lines = 1000 if args.mode == "preview" else 0
    if (args.mode == "preview") != (args.preview_lines > 0) or args.preview_lines < 0:
        parser.error("A positive --preview-lines is only permitted for preview.")
    if not 0 <= args.sample_size <= 10_000:
        parser.error("Sample size must be 0 through 10000.")
    if args.expected_records is not None and args.expected_records <= 0:
        parser.error("--expected-records must be positive.")
    if args.min_free_gib is None:
        args.min_free_gib = 2 if args.mode in ("full", "plan") else 0.05
    if not 0 <= args.min_free_gib < 1e6:
        parser.error("Invalid --min-free-gib.")
    return args


def run(args):
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    status_path = folder / "RUN_STATUS.json"
    database = folder / "_working.sqlite"
    status = dict(status="running", stage="preflight", started_utc=structure.utc_now(), run_id=uuid4().hex,
                  mode=args.mode, active_input=None, processed_inputs=[])
    write_json(status_path, status)
    manifest = dict(**status, inputs={}, scope=None)
    index = None
    started = time.monotonic()
    try:
        project_root = Path(__file__).resolve().parent.parent
        research = load_research_contract(project_root)  # fail closed on question drift
        config, root, unit, config_hash = load_config(args.config)
        role = config.get("input_role", "dataset")
        if role not in ("dataset", "sample", "synthetic") or (args.mode == "synthetic") != (role == "synthetic"):
            raise ValueError("Synthetic mode requires input_role=synthetic, which is exclusive to this mode.")
        datasets = structure.collect_dataset_paths(config, root)
        input_role_contract = validate_dataset_roles(datasets, config)
        scope = ("synthetic_fixture" if args.mode == "synthetic" else "head_preview"
                 if args.mode == "preview" else "sample_files" if role == "sample"
                 else "entire_supplied_dataset_files")
        sequence = args.mode == "synthetic" or (args.mode == "full" and role == "dataset")
        files = {cat: paths["reviews"].name for cat, paths in datasets.items()}
        order = sorted(files, key=lambda cat: (files[cat], cat))
        write_json(folder / "RESEARCH_CONTRACT_E01.json", research)
        (folder / "RESEARCH_CONTRACT_E01_FA.md").write_text(research_markdown(research, scope), encoding="utf-8")
        manifest.update(input_role_contract=input_role_contract,
                        research_contract=dict(requirement="E01", version=research["contract_version"],
                        filename="RESEARCH_CONTRACT_E01.json", source_card_sha256=research["source_document_sha256"],
                        phase1_is_descriptive=True),
                        scope=scope, timestamp_unit=unit, config_sha256=config_hash, config_path=str(args.config.resolve()),
                        input_role=role, review_file_names=files, source_provenance_as_configured=config.get("source_provenance", {}),
                        code_versions=dict(eda=VERSION, structure=structure.VERSION, quality=quality.VERSION),
                        code_sha256=source_manifest(), runtime=dict(python=platform.python_version(),
                        sqlite=sqlite3.sqlite_version, platform=platform.platform(),
                        matplotlib=None if args.mode == "plan" else version("matplotlib")),
                        sequence_status="provisional" if sequence else "suppressed",
                        sampling=dict(seed=args.seed, product_cap_per_primary_category=args.sample_size,
                                      raw_statistics="no subsampling within scanned scope"),
                        exactness=dict(integer_counts="exact for defined scanned populations",
                                       numeric_moments="floating-point population summaries",
                                       log_quantiles="legacy approximate bin bounds unchanged",
                                       exact_numeric="SQLite on-disk frequencies for valid IEEE754-representable values; nearest rank",
                                       exact_outliers="Tukey 1.5*IQR from exact nearest-rank Q1/Q3; strict outside",
                                       exact_nonrepresentable="explicitly withheld rather than rounding") ,
                        provisional_decisions=[
                            "Candidate versions are distinct canonical objects with matching usable user, raw parent, raw epoch; not proof of duplicate identity.",
                            "Main conflict policy: "+args.conflict_policy+"; the other policy is a sensitivity analysis.",
                            "Entire supplied files bound observation; no approved temporal split or reliable follow-up cutoff.",
                            "Nonblank text is optional; descriptor syntax does not prove media accessibility.",
                            "Full means entire local supplied files, not proof of upstream release equality."])
        plan = estimate(datasets, folder, args.mode, args.preview_lines, args.expected_records, args.min_free_gib)
        manifest["disk_plan"] = plan
        write_json(folder / "RESOURCE_PLAN.json", plan)
        if args.mode == "plan":
            status.update(status="planned", stage="done")
        else:
            floor = plan["enforced_free_space_floor_bytes"]

            def check_space():
                if shutil.disk_usage(folder).free < floor:
                    raise OSError("Free space fell below the configured minimum.")

            check_space()
            index = Index(database, sequence)
            manifest["resource_observations"] = {
                "status": "sampled_at_input_boundaries_only",
                "unit": "bytes",
                "observations": [],
                "warning": "Not a peak-RSS or total-scratch-space upper bound; pilot and instrument real run."}
            raw, relations = {}, {}
            print(f"Scope: {scope} | Output: {folder}", flush=True)
            for cat in order:
                accumulator, associations = RawCategory(unit), ReviewRelations()
                accumulator.bind_exact_numeric(index.exact_numeric, cat)
                manifest["inputs"][cat] = {}
                for file_role, path in datasets[cat].items():
                    status.update(stage="scan", active_input=dict(category=cat, role=file_role, path=str(path)))
                    write_json(status_path, status)
                    print(f"[{cat}/{file_role}] reading", flush=True)

                    def consume(row):
                        if file_role == "reviews":
                            parent, user, stamp, star, title, body, length, votes, verified, text, year = accumulator.review(row)
                            associations.add(star, title, body, length, votes, verified, year,
                                image_available=quality.nested_has_usable_content(row.get("images"),"review_image_http_url"))
                            index.review(cat, row, parent, user, stamp, star, text, year)
                        else:
                            index.meta(cat, row, *accumulator.meta(row))

                    def progress(n):
                        if n % config["batch_size"] == 0:
                            index.flush()
                            check_space()
                        interval = config["progress_every_lines"]
                        if interval and n % interval == 0:
                            status["active_input"]["lines_processed"] = n
                            write_json(status_path, status)
                            print(f"[{cat}/{file_role}] {n:,} physical lines", flush=True)

                    info = scan(path, args.preview_lines, consume, progress)
                    index.flush()
                    check_space()
                    manifest["inputs"][cat][file_role] = info
                    manifest["resource_observations"]["observations"].append({
                        "category": cat, "role": file_role,
                        "sqlite_main_file_bytes": database.stat().st_size,
                        "free_disk_bytes": shutil.disk_usage(folder).free,
                        "parsed_objects_in_input": info["counts"]["objects"]})
                    status["processed_inputs"].append(dict(category=cat, role=file_role, counts=info["counts"],
                                                           scanned_content_sha256=info["scanned_content_sha256"]))
                raw[cat] = accumulator.result()
                # Keep exact raw metrics and dedup-scope sensitivity separate.
                precision = index.exact_numeric_profiles(cat,raw[cat])
                raw[cat]['reviews']['exact_numeric'] = precision['reviews']
                raw[cat]['metadata']['exact_numeric'] = precision['metadata']
                raw[cat]['metadata']['dedup_numeric_sensitivity_e09'] = index.metadata_numeric_sensitivity(cat)
                raw[cat]["metadata"]["categorical"] = {
                    key:index.nominal_profile(cat,key,accumulator.metadata)
                    for key in index.nominal_fields(cat)}
                raw[cat]["reviews"]["reviewer_activity"] = index.user_volume_profile(cat,accumulator.reviews)
                relations[cat] = associations.result(accumulator.reviews)
            status.update(stage="index_and_aggregate", active_input=None)
            write_json(status_path, status)
            index.prepare()
            check_space()
            products = {cat: raw_product_results(index, cat, raw[cat]["reviews"]["total"]) for cat in order}
            policies = (args.conflict_policy, next(p for p in POLICIES if p != args.conflict_policy))
            global_summary, cohorts = reconcile(
                index, files, policies, args.sample_size, args.seed, unit,
                core_star_counts_raw={cat: raw[cat]["reviews"]["valid_core_star_counts_raw"]
                                      for cat in order} if sequence else None)
            global_summary['core_numeric_exact_duplicate_sensitivity_e09'] = (
                index.global_core_numeric_sensitivity())
            manifest["quality_evidence"] = reconcile_quality(
                manifest["inputs"], raw, scope, unit, directory=args.quality_dir,
                archive=args.quality_archive)
            manifest["sqlite_bytes_before_cleanup"] = database.stat().st_size
            index.db.close()
            index = None
            database.unlink()
            status.update(stage="exports", active_input=None)
            write_json(status_path, status)
            export(folder, manifest["inputs"], products, raw, relations, cohorts, global_summary, scope, args.conflict_policy)
            status.update(status="complete", stage="done")
        status.update(completed_utc=structure.utc_now(), elapsed_seconds=round(time.monotonic()-started, 3))
        manifest.update(status)
        write_json(folder / "manifest.json", manifest)
        write_json(status_path, status)
        if status['status'] == 'complete':
            # Save the exact input configuration and a self-verifiable artifact
            # inventory AFTER manifest/RUN_STATUS finalization. Neither is an
            # interpretation of the underlying data.
            shutil.copyfile(args.config, folder / 'RUN_CONFIG_SNAPSHOT.json')
            write_inventory(folder)
    except BaseException as exc:
        if index is not None:
            index.db.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            database.with_name(database.name+suffix).unlink(missing_ok=True)
        status.update(status="failed", ended_utc=structure.utc_now(), error_type=type(exc).__name__)
        manifest.update(status)
        write_json(folder / "manifest.json", manifest)
        write_json(status_path, status)
        raise
    print(f"Status: {status['status']} | {folder}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, EOFError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"EDA failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
