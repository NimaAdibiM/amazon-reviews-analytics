"""Write separate evidence for E09, E26 and E27 without changing legacy figures."""
from .report import markdown, write_csv


def _fmt(v):
    if v is None:
        return '—'
    return f'{v:.8g}' if isinstance(v, float) else str(v)


def write_precision(folder, raw, global_summary):
    lines = ['# E09 / E26 / E27 — شواهد فراوانی دقیق و حساسیت به کپی', '',
       'این خروجی فقط برای **دامنهٔ ورودی همین اجرا** محاسبه شده است؛ اجرای مصنوعی یا پیش‌نمایش، شاهد جمعیت واقعی نیست.',
       'روش صدک دقیق: آمارهٔ مرتب‌شده در رتبهٔ `ceil(p × n)` (با رتبه‌گذاری از ۱). '
       'این روش با صدک درون‌یابی‌شدهٔ خطی B یکسان نیست. تعریف قبل از مقایسه باید ثابت باشد.',
       'پرت: کمتر از چارک اول منهای ۱٫۵ برابر فاصلهٔ بین چارکی، یا بیشتر از چارک سوم به‌علاوهٔ ۱٫۵ برابر آن. '
       'مرزها **سخت‌گیرانه** هستند؛ مساوی مرز پرت نیست. هیچ داده‌ای خودکار حذف نمی‌شود.',
       'فراوانی‌های دقیق در SQLite روی دیسک تجمیع می‌شوند؛ مقادیر صحیح بزرگ‌تر از ۲^۵۳ '
       'به‌جای گردشدن، از استنتاج «دقیق» خارج و در دلیل عدم امکان ثبت می‌شوند.',
       'خروجی‌های بازه‌ای نمودارهای قدیمی حفظ شده‌اند و باید با این فایل مکمل خوانده شوند.', '',
       '## محدودیت E09',
       'حساسیت نقدها فقط برای **هستهٔ معتبر توالی** با شناسهٔ والد، زمان و امتیاز معتبر است '
       'و حذف کپی در سطح تمام چهار دسته و با برابری کاننیکال رکورد انجام شده است. '
       'حساسیت فراداده محدود به رکوردهای دارای شناسهٔ والد معتبر و حذف کپی در همان دسته/محصول است. '
       'این نتایج معادل حساسیت تمام رکوردهای خام و تمام متغیرها نیست؛ E09 تا گسترش آن باز می‌ماند.', '']
    allrows=[]
    for cat, data in sorted(raw.items()):
        lines += [f'## {cat} — E26/E27', '',
                  '| نقش | شاخص | شمار معتبر | وضعیت دقت | میانه | چارک ۱ | چارک ۳ | پرت دقیق |',
                  '|---|---|---:|---|---:|---:|---:|---:|']
        rows=[]
        for role in ('reviews','metadata'):
            for metric, p in data[role]['exact_numeric'].items():
                q=p['quantiles'] or {}; t=p['tukey'] or {}
                rows.append((role,metric,p['expected_valid_observations'],
                             p['status'],q.get('p05'),q.get('p25'),q.get('p50'),
                             q.get('p75'),q.get('p95'),q.get('p99'),q.get('p999'),
                             p['iqr'],t.get('lower_fence'),t.get('upper_fence'),
                             t.get('below_count'),t.get('above_count'),t.get('total_outlier_count'),
                             t.get('denominator'),t.get('fraction'),str(p['rejected_for_exact'])))
                lines.append(f"| {role} | `{metric}` | {p['expected_valid_observations']} | "
                             f"{p['status']} | {_fmt(q.get('p50'))} | {_fmt(q.get('p25'))} | "
                             f"{_fmt(q.get('p75'))} | {_fmt(t.get('total_outlier_count'))} |")
        write_csv(folder/cat/'E26_E27_EXACT_NUMERIC.csv',
                  ('role','metric','valid','precision_status','p05','p25','p50','p75',
                   'p95','p99','p999','iqr','lower_fence','upper_fence',
                   'below_count','above_count','tukey_outlier_count','tukey_denominator',
                   'tukey_fraction','unsupported_values'),rows)
        lines += ['', '### E09 — فراداده با والد معتبر', '',
                  '| سنجه | خام | پس از حذف کپی | اختلاف | فاصلهٔ تغییرکل توزیع |',
                  '|---|---:|---:|---:|---:|']
        meta=data['metadata']['dedup_numeric_sensitivity_e09']
        erows=[]
        for metric,d in meta['metrics'].items():
            a=d['raw']; b=d['exact_unique']
            erows.append((metric,a['expected_valid_observations'],b['expected_valid_observations'],
                         d['removed_valid_observations'],d['total_variation_distance'],
                         a['status'],b['status'],
                         (a['quantiles'] or {}).get('p50'),(b['quantiles'] or {}).get('p50')))
            lines.append(f"| `{metric}` | {a['expected_valid_observations']} | "
                         f"{b['expected_valid_observations']} | {d['removed_valid_observations']} | "
                         f"{_fmt(d['total_variation_distance'])} |")
        write_csv(folder/cat/'E09_METADATA_SENSITIVITY.csv',
                  ('metric','raw_valid_n','unique_valid_n','removed_valid_n','total_variation',
                   'raw_status','unique_status','raw_median','unique_median'),erows)
        lines += ['']
    g=global_summary['core_numeric_exact_duplicate_sensitivity_e09']
    lines += ['## E09 — هستهٔ توالی معتبر سراسری', '',
              f"وضعیت: {g['status']}؛ دامنه: {g.get('scope','—')}", '',
              '| سنجه | خام | یکتا | فاصلهٔ تغییرکل توزیع |',
              '|---|---:|---:|---:|']
    grow=[]
    for metric,d in g.get('metrics',{}).items():
        a,b=d['raw'],d['exact_unique']
        grow.append((metric,a['expected_valid_observations'],b['expected_valid_observations'],
                    d['removed_valid_observations'],d['total_variation_distance'],
                    a['status'],b['status'],
                    (a['quantiles'] or {}).get('p50'),(b['quantiles'] or {}).get('p50')))
        lines.append(f"| `{metric}` | {a['expected_valid_observations']} | "
                     f"{b['expected_valid_observations']} | {_fmt(d['total_variation_distance'])} |")
    write_csv(folder/'E09_GLOBAL_CORE_SENSITIVITY.csv',
              ('metric','raw_valid_n','unique_valid_n','removed_valid_n','total_variation',
               'raw_status','unique_status','raw_median','unique_median'),grow)
    lines += ['', '## وضعیت پذیرش', '',
              '- E26: پیاده‌سازی محاسبهٔ دقیق تکمیل شد؛ نیازمند اجرای کامل واقعی و ممیزی دقت/هزینه.',
              '- E27: قاعده و شمارش پرت‌ها دقیق هستند، ولی تفسیر علمی دادهٔ واقعی و تعریف نامعتبرها نیازمند بازبینی است.',
              '- E09: جمعیت‌های حساسیت صریح و قابل‌آزمون‌اند، اما هنوز همهٔ رکوردها/سنجه‌ها پوشش ندارند.',
              '- وقتی بزرگی یک عدد مانع دقت IEEE-754 شود، نتیجهٔ دقیق منتشر نمی‌شود.', '']
    markdown(folder/'PRECISION_E09_E26_E27_FA.md',lines)
