"""Write archived, explicitly non-EDA Q counts for four-category comparison.

This only processes original Q01–Q25 summaries; never rebrands them as a
fresh raw-file EDA or an independent test of missingness mechanisms.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .quality_reference import summarize_archive
from eda.quality_link import _extract_archive


def build(archive, output):
    archive = Path(archive)
    output = Path(output)
    root = Path(__file__).resolve().parents[1]
    verification = summarize_archive(archive, root)
    summaries = _extract_archive(archive)
    rows = []
    field_rows = []
    for cat, (summary, digest) in summaries.items():
        rv = summary['files']['reviews']; md = summary['files']['metadata']
        r, m = rv['object_records'], md['object_records']
        rmod, mmod = rv['modality_availability'], md['modality_availability']
        def count(info, key):
            n = info[key]['rows']
            if type(n) is not int or n < 0:
                raise ValueError(f'Invalid Q metric {cat}:{key}')
            return n
        valid_stars = rv['q11_review_rating_validity']['valid_integral_stars_rows']
        classes = rv['q11_review_rating_validity']['valid_star_counts']
        low = classes['1'] + classes['2']
        duplicate_extra = rv['q04_review_exact_duplicates']['extra_duplicate_review_rows']
        fields = dict(
            category=cat, review_records=r, metadata_records=m,
            valid_rating_count=valid_stars, low_rating_count=low,
            low_share_valid_rating=low/valid_stars if valid_stars else None,
            nonblank_body_count=count(rmod,'nonblank_review_body'),
            nonblank_body_share_review=count(rmod,'nonblank_review_body')/r if r else None,
            review_image_url_count=count(rmod,'image_http_url_reference'),
            review_image_url_share_review=count(rmod,'image_http_url_reference')/r if r else None,
            valid_current_price_count=count(mmod,'valid_current_price'),
            valid_current_price_share_metadata=count(mmod,'valid_current_price')/m if m else None,
            product_image_url_count=count(mmod,'image_http_url_reference'),
            product_image_url_share_metadata=count(mmod,'image_http_url_reference')/m if m else None,
            exact_duplicate_extra_review=duplicate_extra,
            exact_duplicate_extra_share_review=duplicate_extra/r if r else None,
            quality_summary_sha256=digest)
        for numerator,denom in ((valid_stars,r),(low,valid_stars),(duplicate_extra,r),
                                (fields['nonblank_body_count'],r),(fields['review_image_url_count'],r),
                                (fields['valid_current_price_count'],m),(fields['product_image_url_count'],m)):
            if not 0 <= numerator <= denom:
                raise AssertionError(f'Bad Q count in {cat}')
        if sum(classes[str(i)] for i in range(1,6)) != valid_stars:
            raise AssertionError(f'Q star categories do not reconcile in {cat}')
        rows.append(fields)
        for role, qrole in (("reviews", rv), ("metadata", md)):
            for field, status in qrole.get("q02_field_missingness", {}).items():
                components = ("absent", "null", "blank_string", "empty_array", "empty_object", "present_nonempty")
                present = {name: status[name + "_rows"] for name in components}
                if sum(present.values()) != qrole["object_records"]:
                    raise AssertionError(f"Q02 raw field states do not conserve records: {cat}/{role}/{field}")
                field_rows.append(dict(category=cat, role=role, field=field,
                                       denominator=qrole["object_records"],
                                       **present,
                                       missing_or_empty_count=sum(present[k] for k in components[:-1]),
                                       missing_or_empty_fraction=(sum(present[k] for k in components[:-1])/
                                                                  qrole["object_records"] if qrole["object_records"] else None)))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'REAL_Q_FIELD_MISSINGNESS.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        if field_rows:
            w=csv.DictWriter(handle,fieldnames=list(field_rows[0]));w.writeheader();w.writerows(field_rows)
    with (output/'REAL_Q_CATEGORY_COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    obj = {
        'status':'verified_archived_quality_summaries_only',
        'source_quality_archive_sha256':verification['archive_file_sha256'],
        'source_quality_run_id':verification['source_run_id'],
        'records':rows,
        'q02_field_state_rows_exported':len(field_rows),
        'limitations':['No eight original raw JSONL files were uploaded for independent EDA validation.',
                       'Product image URL is not proof that an image was downloaded.',
                       'Differences across categories do not prove a mechanism of missingness.',
                       'Source category members may overlap at parent-product level.',
                       'Raw review and metadata records use distinct denominators.']}
    (output/'REAL_Q_CATEGORY_COMPARISON.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines = ['# چهار دسته — شاهد واقعی از Q01–Q25 (نه اجرای مجدد EDA)', '',
             f"شناسهٔ اجرای مرجع: `{verification['source_run_id']}`.", '',
             '| دسته | نقدها | متن غیرخالی از نقدها | تصویر نقد از نقدها | قیمت معتبر از فراداده | تصویر محصول از فراداده | امتیاز ۱–۲ از امتیازهای معتبر | کپی دقیق اضافی از نقدها |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        def pct(v): return 'ناموجود' if v is None else f'{v:.2%}'
        lines.append('| '+row['category']+' | '+f"{row['review_records']:,}"+' | '+ ' | '.join(pct(row[k]) for k in (
            'nonblank_body_share_review','review_image_url_share_review','valid_current_price_share_metadata',
            'product_image_url_share_metadata','low_share_valid_rating','exact_duplicate_extra_share_review'))+' |')
    lines += ['','**حدود استنباط:** این اعداد از گزارش کیفیت انجام‌شده روی دادهٔ واقعی استخراج شده‌اند؛ هیچ‌کدام خروجی اجرای EDA جدید روی فایل خام نیست. مقایسهٔ چهار دسته توصیفی است، از درصدها نمی‌توان تصادفی‌بودن گمشدگی را نتیجه گرفت. اندازهٔ دسته‌ها و زمان‌های جمع‌آوری ممکن است متفاوت باشند.']
    (output/'REAL_Q_CATEGORY_COMPARISON_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return obj


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args(); result=build(args.archive,args.out)
    print('Archived Q comparison written:', len(result['records']), 'categories')

if __name__=='__main__':main()
