"""Read prior real Q01–Q25 archive ONLY; never claim an EDA raw rerun."""
import argparse,csv,json
from pathlib import Path
from zipfile import ZipFile

CATEGORIES=('All_Beauty','Appliances','Beauty_and_Personal_Care','Electronics')

def build(source, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ZipFile(source) as z:
        assert z.testzip() is None
        for cat in CATEGORIES:
            matches=[x for x in z.namelist() if f'/full/{cat}/' in x and x.endswith('/quality_summary.json')]
            if len(matches)!=1:raise ValueError(f'{cat}: {len(matches)} quality summaries')
            item=json.loads(z.read(matches[0]))
            if item['scope']!='entire_supplied_dataset_files' or item['source_category']!=cat:
                raise ValueError('unexpected scope/category in quality archive')
            reviews=item['files']['reviews']['object_records']
            q22=item['q22_review_text_lengths'];q23=item['q23_review_short_long_text'];q24=item['q24_review_unicode_audit']
            if not (q22['denominator_parsed_review_objects']==reviews==q23['denominator_parsed_review_objects']==q24['denominator_parsed_review_objects']):raise ValueError('denominator mismatch')
            row=dict(category=cat,parsed_review_objects=reviews,
                     valid_star_rows=item['files']['reviews']['q11_review_rating_validity']['valid_integral_stars_rows'],
                     body_string_rows=q22['string_rows'],
                     nonblank_body_rows=q22['states']['nonblank_string'],
                     blank_body_rows=q22['states']['empty_or_whitespace_only_string'],
                     median_words_all_strings=q22['distributions']['whitespace_words']['quantiles']['p50'],
                     median_chars_all_strings=q22['distributions']['unicode_codepoints']['quantiles']['p50'],
                     short_nonblank_at_most_3_words=q23['primary_short_nonblank_rows'],
                     short_denominator_nonblank=q23['nonblank_body_rows'],
                     long_all_string_over_10000_codepoints=q23['primary_long_string_rows'],
                     long_denominator_strings=q23['string_valued_body_rows'],
                     multiple_letter_script_texts=q24['pattern_counts']['multiple_letter_scripts'],
                     replacement_char_rows=q24['flag_counts']['replacement_character'],
                     valid_metadata_prices=item['files']['metadata']['modality_availability']['valid_current_price']['rows'],
                     metadata_objects=item['files']['metadata']['object_records'])
            rows.append(row)
    path=output/'REAL_Q_UNIVARIATE_REFERENCE.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# شواهد واقعی Q01 تا Q25 مرتبط با E14–E30','',
           '**منبع:** ZIP ممیزی کیفیت کاربر؛ این خروجی اجرای مجدد EDA روی دادهٔ خام نیست. '
           'اعداد قبل از حذف تکراری‌ها و با دامنهٔ تمام فایل‌های تأمین‌شده گزارش شده‌اند.','',
           '| دسته | کل نقد خام | رشتهٔ بدنه | میانهٔ واژه‌های متن | کوتاه ≤۳ واژه | بلند بیش از ۱۰٬۰۰۰ نویسه | چند اسکریپت |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["category"]} | {r["parsed_review_objects"]:,} | {r["body_string_rows"]:,} | '
                     f'{r["median_words_all_strings"]:g} | {r["short_nonblank_at_most_3_words"]:,} | '
                     f'{r["long_all_string_over_10000_codepoints"]:,} | {r["multiple_letter_script_texts"]:,} |')
    lines+=['', 'مخرج کوتاه «متن رشته‌ای ناتهی» و مخرج بلند «تمام متن‌های رشته‌ای» است. '
             'وجود چند اسکریپت تشخیص زبان نیست؛ برآورد زبان مستقل و نسخه‌دار نیاز است.','',
             'برای اتصال این شواهد به EDA تازه باید هر هشت اثر انگشت محتوایی و دامنه و تعریف شاخص با فایل‌های خام واقعی یکی باشد. '
             'چون فایل‌های خام اینجا موجود نیستند، این گزارش فقط از نظر سازگاری داخلی آزموده شده است.','']
    (output/'REAL_Q_UNIVARIATE_REFERENCE_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (output/'REAL_Q_UNIVARIATE_REFERENCE.json').write_text(json.dumps({'source':'archived real quality run','scope':'reference_only_no_raw_eda','rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();build(args.source,args.out)
