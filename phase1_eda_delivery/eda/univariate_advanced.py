"""E14--E30: additional read-only, bounded-memory descriptive evidence.

No language inference is made from Unicode script membership. The archived
Q24 analysis is only attributable to this EDA after the entire raw input
fingerprint and the Q definition agree.
"""
from . import plots
from .report import markdown, percent, distribution_text, write_json, write_csv


def _label(text):
    # Escape only Markdown syntax; never clean or harmonize source labels.
    return str(text).replace('\\','\\\\').replace('|','\\|').replace('\n',' ')


def export_extended(folder, category, raw, scope):
    rev, meta=raw['reviews'], raw['metadata']
    n=rev['numeric']; m=meta['numeric']
    b=rev['rating_balance']
    lines=[f'# {category} — دفتر شواهد تک‌متغیره E14–E30','',
           f'دامنه: `{scope}`؛ تمام رکوردهای تکراریِ خام حفظ شده‌اند. هیچ نتیجهٔ مدل‌سازی از این گزارش استخراج نمی‌شود.','',
           '## E14–E15: پنج ردهٔ امتیاز و عدم‌توازن',
           'واحد: یک رکورد نقد با امتیاز صحیح ۱ تا ۵؛ نمرهٔ نامعتبر از مخرج خارج و در `numeric.rating.excluded` گزارش شده است.','',
           '| ستاره | تعداد | سهم از امتیاز معتبر |','|---|---:|---:|']
    for star, x in b['all_five_classes'].items():
        lines.append(f'| {star} | {x["numerator"]:,} | '+(f'{x["fraction"]:.4%}' if x['fraction'] is not None else 'تعریف‌نشده')+' |')
    lines+=['',f'۱ یا ۲ ستاره: {percent(b["low_1_2"])}؛ ۳ ستاره: {percent(b["neutral_3"])}؛ '
                f'۴ یا ۵ ستاره: {percent(b["high_4_5"])}.',
            f'رده‌های غایب: {b["missing_classes"]}؛ نسبت بیشینه به کمینهٔ پنج رده: {b["max_to_min_all_five_ratio"]}؛ '
            f'آنتروپی نرمال‌شده نسبت به پنج رده: {b["normalized_entropy_five_classes"]}.',
            'این نسبت‌ها برچسب ده نقد بعدی در سطح محصول نیستند.','',
            '## E16–E17: طول متن و آستانه‌های Q22/Q23',
            'معیار Unicode در این گزارش تعداد code point پایتون است، و تعداد واژه با split بر فاصلهٔ سفید به دست می‌آید. '
            'در نمایش «همهٔ رشته‌ها» حتی رشتهٔ خالی یا فقط فاصله حضور دارد و صفر واقعی طول است؛ '
            'مقادیر null/غایب/نوع غلط در مخرج رشته‌ها نیستند. نمایش قدیمی «ناتهی» برای سازگاری باقی مانده است.','']
    for key in ('title_chars_all_strings','title_words_all_strings','body_chars_all_strings','body_words_all_strings'):
        lines += [f'- `{key}`: {distribution_text(n[key])}']
    lines+=['',f'کوتاه با معیار Q23: حداکثر ۳ واژه و بدنهٔ ناتهی، {percent(rev["text_thresholds_q23"]["short_nonblank_words_at_most_3"])}. '
                f'بلند با معیار Q23: بیش از ۱۰۰۰۰ code point، {percent(rev["text_thresholds_q23"]["long_all_string_codepoints_more_than_10000"])}.',
            'مخرج کوتاه، بدنه‌های ناتهی است؛ مخرج بلند، تمام بدنه‌های رشته‌ای (حتی خالی) است. شمارها در هر دو حالت بر حسب اشیای خام گزارش می‌شوند.' ,'',
            '## E18: نویسه/اسکریپت/زبان',
            'پرچم‌های raw Unicode در JSON اصلی ثبت شده‌اند، اما تشخیص زبان از روی اسکریپت مجاز نیست. '
            'نسخهٔ B به fastText بیرونی وابسته است؛ تا اجرای کامل نسخه‌دار یا اتصال سخت‌گیرانهٔ خروجی Q24، '
            'زبان محتمل در این اجرای EDA «محاسبه‌نشده» است، نه «انگلیسی».','',
            '## E19–E22: رأی مفید، خرید، زمان و فراداده','']
    for key in ('helpful_vote','price','average_rating','rating_number'):
        obj=n[key] if key=='helpful_vote' else m[key]
        lines += [f'- `{key}`: {distribution_text(obj)}؛ '
                  f'چولگی لحظه‌ای: {obj["skewness_population"]}؛ کشیدگی اضافی: {obj["excess_kurtosis_population"]}؛ '
                  f'بازهٔ شمار بیش از صدک ۹۵ بر اساس طبقات: {obj["above_p95_count_bounds"]}.']
    lines+=['', '## E08: شش حالت انحصاری محتوای هر فیلد (مطابق Q01)',
            'صورت هر حالت تعداد رکوردهای همان فایل و مخرج تعداد کل اشیای خوانده‌شده است. '
            'مقدار صفر و False از نوع محتوای موجودند، نه گمشده. وضعیت محتوا، آزمون اعتبار دامنه نیست.', '',
            '| نقش | فیلد | غایب | null | رشته خالی | آرایه خالی | شیء خالی | حاضر |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for role,part in (("reviews",rev),("metadata",meta)):
        for field, states in part["field_states_q01"].items():
            counts={x["state"]:x["numerator"] for x in states}
            lines.append(f'| {role} | `{_label(field)}` | '+ ' | '.join(str(counts[x]) for x in
                ("absent","null","blank_string","empty_array","empty_object","present_nonempty"))+' |')
    lines += ['',f'خرید تأییدشده/نشده/نامعتبر: {[(r["level"],r["numerator"]) for r in rev["verified_states"]]}.',
              'رأی مفید تجمعی، به سن نقد و مواجهه وابسته است؛ صفر رأی به‌معنای بی‌کیفیتی متن نیست.',
              'قیمت، میانگین امتیاز و تعداد امتیاز فراداده، snapshot زمان خزش هستند و ویژگی تاریخیِ لحظهٔ نقد پنجم نیستند.',
              f'ماه‌های مشاهده‌شده: {len(rev["months"])}، شامل ماه‌های بدون مشاهده در فاصلهٔ ثبت‌شده. '
              'دورهٔ ابتدا و انتها ممکن است ناقص باشد؛ شمار نقد، فروش یا نرخ تعدیل‌شده با مواجهه نیست.','',
              '## E23 و E28: متغیرهای اسمی و مسیر دسته‌بندی فراداده','',
              'برای `categories.level_k` هر مسیر در موقعیت مشخص شمارش می‌شود؛ '
              '`categories.full_path` فقط مسیر کاملِ دارای فهرست ناتهی از رشته‌های ناتهی را می‌شمارد. '
              'ترتیب حفظ می‌شود و رشتهٔ مسیر کامل به‌صورت آرایهٔ JSON نمایش داده می‌شود.','',
              'حالات ساختار مسیر دسته‌بندی (مخرج: تمام اشیای فراداده): '+
              str({k:v['numerator'] for k,v in meta['category_path_states'].items()}),'']
    for field, val in meta['categorical'].items():
        lines += [f'### `{field}`', '',
                  f'برچسب معتبر: {val["valid"]:,} از {meta["total"]:,} رکورد؛ '
                  f'سطوح متمایز: {val["distinct_levels"]:,}؛ دیگر: {val["other"]["numerator"]:,} رکورد در '
                  f'{val["other"]["levels"]:,} سطح. برچسب خام نرمال‌سازی نشده است.','',
                  f'آنتروپی شانون (طبیعی): {val["shannon_entropy_nats"]}؛ '
                  f'آنتروپی نرمال‌شده: {val["normalized_shannon_entropy"]}؛ '
                  f'غلظت سیمپسون: {val["simpson_concentration"]}؛ '
                  f'تعداد سطح مؤثر: {val["effective_level_count"]}. '
                  'همهٔ برچسب‌های معتبر، نه فقط موارد Top-12، در محاسبه حضور دارند.','',
                  '| برچسب خام | تعداد | سهم از معتبر |','|---|---:|---:|']
        for x in val['top']:
            lines.append(f'| {_label(x["label"])} | {x["numerator"]:,} | '+
                         (f'{x["fraction"]:.3%}' if x['fraction'] is not None else '—')+' |')
        lines.append('')
    lines += ['`main_category` معادل عضویت دستهٔ فایل دانلودشده نیست؛ فروشگاه‌ها نیز الزاماً شناسهٔ یکتای فروشنده نیستند.','',
              '## E24: تعداد ورودی‌ها در برابر محتوای قابل‌استفاده','',
              'شمار آرایه‌ها، تعداد *ورودی‌های سطح اول* است، نه شمار عکس‌های واقعاً قابل‌دانلود؛ '
              'فیلدهای `features_chars` و `description_chars` فقط زمانی معتبرند که همهٔ اعضای آرایه رشته باشند.','']
    for key in ('review_image_entries',):
        lines += [f'- `{key}`: {distribution_text(n[key])}']
    for key in ('features_entries','features_chars','description_entries','description_chars','images_entries',
                'videos_entries','categories_entries','bought_together_entries','details_keys'):
        lines += [f'- `{key}`: {distribution_text(m[key])}']
    lines += ['', 'وجود محتوای واقعی از `reviews.modalities` و `metadata.modalities` با قواعد جداگانه قابل بررسی است.','',
              '## E25: تمرکز نقد در سطح کاربر','',
              f'یک شناسهٔ کاربر یکتا در این دسته یک مشاهده است؛ {rev["reviewer_activity"]["reviewer_count"]:,} کاربر معتبر وجود دارد.',
              f'توزیع نقد به ازای کاربر: {distribution_text(rev["reviewer_activity"]["activity_distribution"])}',
              f'یک درصد پرتکرارترین شناسه‌ها: {percent(rev["reviewer_activity"]["top_one_percent_raw_review_share"])} از نقدهای با شناسهٔ معتبر.',
              'هیچ شناسهٔ خام در خروجی انتشار نمی‌یابد. این توصیف فعالیت مشاهده‌شده است، نه فعالیت کل عمر کاربر.','',
              '## E26–E30: مرز استنباط و نمودارها','',
              'میانگین و انحراف معیار به‌صورت لحظه‌ای بر رکوردهای معتبر (مخرج n) محاسبه شده‌اند؛ '
              'برای سنجه‌های توان‌دویی، صدک‌ها **بازه** هستند و نباید با صدک دقیق B اشتباه شوند. '
              'نمای کلاس‌های امتیاز دقیق است. نمودارها در `univariate.png` و `univariate_extended.png` موجودند.',
              'پرت آماری لزوماً خطای داده نیست؛ حذف خودکار انجام نشده است. '
              'همبستگی و علیت از این توزیع‌های حاشیه‌ای نتیجه نمی‌شوند.','',
              '## موارد باز و نیازمند شاهد','',
              '- محاسبهٔ مستقل زبان روی تمامی متون با مدل مشخص و اعتبارسنجی‌شده اجرا نشده است (E18).',
              '- صدک دقیق برای تمام کمیت‌های پرکاردینال بدون ذخیره‌سازی اضافی محاسبه نشده است (E26/E27).',
              '- اجرای کامل این اصلاحات روی هشت فایل واقعی در این محیط انجام نشده است (E14–E30).',
              '- برچسب‌های بلند برخی فروشگاه‌ها در نمودار خلاصه می‌شوند؛ جدول آن‌ها خام است.','']
    markdown(folder/'UNIVARIATE_EXTENDED_FA.md',lines)
    write_json(folder/'univariate_extended.json',
               dict(category=category,scope=scope,rating_balance=b,months=rev['months'],
                    categorical_metadata=meta['categorical'],reviewer_activity=rev['reviewer_activity'],
                    text_definition='all string values, including blank, separate from nonblank views',
                    language_status='not_computed_without_validated_model',
                    quantile_warning='power-of-two distributions provide quantile intervals only'))
    rows=[]
    for name,d in [('reviews.'+k,v) for k,v in n.items()]+[('metadata.'+k,v) for k,v in m.items()]:
        rows.append((name,d['total'],d['valid'],d['excluded'],d['minimum'],d['maximum'],d['mean'],
                     str(d['quantile_bounds'].get('p50')), d['bin_rule'],
                     d.get('skewness_population'),d.get('excess_kurtosis_population'),
                     str(d.get('above_p95_count_bounds'))))
    write_csv(folder/'univariate_extended_numeric.csv',
              ('measure','total','valid','excluded','minimum','maximum','mean','p50_bounds','quantile_method','skewness_population','excess_kurtosis_population','above_p95_count_bounds'),rows)
    write_csv(folder/'univariate_monthly_utc.csv',
              ('month_utc','count','valid_timestamp_denominator','fraction'),
              ((x['month'],x['numerator'],x['denominator'],x['fraction']) for x in rev['months']))
    write_csv(folder/'univariate_category_imbalance.csv',
              ('field','valid_records','excluded_records','distinct_levels','shannon_entropy_nats',
               'normalized_entropy','simpson_concentration','effective_levels'),
              ((field,d['valid'],d['excluded'],d['distinct_levels'],d['shannon_entropy_nats'],
                d['normalized_shannon_entropy'],d['simpson_concentration'],d['effective_level_count'])
               for field,d in meta['categorical'].items()))
    write_csv(folder/'univariate_field_states.csv',
              ('role','field','state','count','parsed_objects','fraction'),
              ((role,field,item['state'],item['numerator'],item['denominator'],item['fraction'])
               for role,part in (("reviews",rev),("metadata",meta))
               for field,states in part['field_states_q01'].items() for item in states))
    write_csv(folder/'univariate_category_path_states.csv',
              ('state','count','metadata_objects','fraction'),
              ((state,obj['numerator'],obj['denominator'],obj['fraction'])
               for state,obj in meta['category_path_states'].items()))
    write_csv(folder/'univariate_nominal_metadata.csv',
              ('field','label','count','valid_label_denominator','fraction','is_other_aggregated'),
              ((field,item['label'],item['numerator'],item['denominator'],item['fraction'],False)
               for field,stats in meta['categorical'].items() for item in stats['top']))
    write_csv(folder/'univariate_reviewer_volume_groups.csv',
              ('volume_bin','users','valid_users_denominator','fraction'),
              ((key,r['numerator'],r['denominator'],r['fraction'])
               for key,r in rev['reviewer_activity']['volume_groups'].items()))
    plots.extended_univariate_plot(folder,category,rev,meta,scope)
