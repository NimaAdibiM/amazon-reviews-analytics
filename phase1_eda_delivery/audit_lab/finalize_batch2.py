"""Build versioned individual verdicts for the second foundation audit batch.

All states are evidence-scoped. Presence of a test is not blanket scientific approval.
"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

# Five independently reviewed fields per requirement; tests are separately identified.
FINDINGS={
'E03':{
 'A':'`eda/input_data.scan` اثر انگشت متن بازشده و فایل فشرده را ثبت می‌کند؛ `eda.__main__` منشأ تنظیم‌شده را به مانیفست می‌افزاید. پیش‌تر کیفیت فقط از پوشه خوانده می‌شد و راستی‌آزمایی شمارش‌های مرتبط انجام نمی‌شد.',
 'B':'`Univariate_EDA.file_sha` هش را می‌گیرد و نام و دامنه را در خروجی ثبت می‌کند؛ ولی فقط چهار ورودی مربوط به دو دسته را پوشش می‌دهد و منشأ upstream را مستقلاً اثبات نمی‌کند.',
 'choice':'ساختار A حفظ شد؛ `eda/quality_link.py` برای خواندن مستقیم ZIP کیفیت، کنترل هش خود بسته، نسخهٔ کدهای مرجع، چهار دامنه و تطبیق اثر انگشت/تعداد/خط فیزیکی اضافه شد. هش محلی هرگز اصالت سرور مبدأ نامیده نمی‌شود.',
 'tests':['test_e03_all_source_hashes_prove_same_local_bytes','test_e03_and_e11_read_actual_Q_archive_without_reinterpreting_as_raw_scan','QUALITY_ARCHIVE_REJECTION'],
 'status':'پوشش و آزمون مصنوعی موفق؛ تطبیق هش هشت فایل خام واقعی باز است.'},
'E04':{
 'A':'حالت‌های `synthetic`، `preview`، `full`، `plan` و برچسب‌های دامنه از قبل وجود داشتند؛ تولید جامعهٔ ۵+۱۰ برای پیش‌نمایش مهار شده است.',
 'B':'`--max-lines` حالت جزئی ایجاد و `complete` را ثبت می‌کند، اما اجرای پیش‌فرض محدود به دو دسته است و برچسب جامعهٔ چهار‌دسته‌ای ندارد.',
 'choice':'حفظ معماری A. یک اجرای مستقل `preview` با فایل نمونه و محافظ ممنوعیت توالی انجام شد؛ خروجی برنامه از پیش‌نمایش نتیجهٔ کامل استنتاج نمی‌کند.',
 'tests':['test_e04_preview_and_incomplete_sequencing_are_separated','tests.test_eda.EndToEnd.test_preview_is_prefix_and_does_not_publish_cohort'],
 'status':'قراردادهای دامنه در دادهٔ مصنوعی تأیید شدند؛ کامل‌بودن اسکن حقیقی بررسی نشده است.'},
'E05':{
 'A':'جامعهٔ نقد، شیء فراداده، محصول درون دسته و محصول جهانی را در `eda/univariate.py` و `eda/products.py` تفکیک کرده است؛ به‌ویژه عضویت دسته‌ای با محصول جهانی متفاوت است.',
 'B':'واحد تعداد نقد/کاربر/والد و فراداده را جدا گزارش می‌کند، ولی محصول مشترک بین چهار دسته و جامعهٔ پیش‌بینی ندارد.',
 'choice':'A حفظ شد. کنترل مستقل جمع معتبر+خارج از دامنه برای هر شاخص و نسبت عضویت‌های دسته به محصولات جهانی اجرا شد. هیچ تعداد نقد با تعداد محصول جایگزین نشد.',
 'tests':['test_e05_units_and_observation_denominators','tests.test_eda.EndToEnd.test_independent_cohort_arithmetic_and_overlap'],
 'status':'همسانی واحدهای تعریف‌شده در اجرای مصنوعی بررسی شد؛ ممیزی همهٔ نمودارهای واقعی باقی است.'},
'E06':{
 'A':'`eda/stats.frequency` مقدار تهی را برای مخرج صفر تولید می‌کرد، اما در صورت ورودی شمارش منفی یا صورت بزرگ‌تر از مخرج خطای صریح نداشت.',
 'B':'`pct` و فراوانی‌ها مخرج‌های معتبر را جدا می‌کنند؛ اما سازوکاری برای رد فوری شمارش ناممکن در همهٔ خروجی‌ها ندارد.',
 'choice':'به `frequency` کنترل نوع صحیح و بازهٔ ۰ ≤ صورت ≤ مخرج افزوده شد. نرخ‌های تمام خروجی‌های مصنوعی با محاسبهٔ مستقل بازبینی و ۰/۰ تهی نگه داشته شد.',
 'tests':['test_e06_all_rate_numerators_and_denominators','test_e06_frequency_raises_on_impossible_counts','audit_lab.tests.test_batch1.FoundationalOracles.test_frequency_zero_denominator_is_undefined'],
 'status':'در جامعهٔ مصنوعی آزموده شد؛ دفتر نرخ‌های همهٔ خروجی‌های اجرای واقعی تولید نشده است.'},
'E07':{
 'A':'از parser مرجع Q با UTF-8 سخت‌گیرانه، رد کلید تکراری، NaN/بی‌نهایت و شمارش خطاهای خواندن استفاده می‌کند.',
 'B':'`parse_line` نیز سخت‌گیرانه است، اما طبقه‌بندی خطاهایش متفاوت و شمارش به چهار فایل ثابت دو دسته محدود است.',
 'choice':'منطق Q ثابت ماند. موارد کلید تکراری، عدد نامتناهی، JSON خراب، شیءنبودن، خط خالی و UTF-8 نامعتبر با پاسخ معلوم آزمایش شدند.',
 'tests':['test_e07_malformed_json_and_unicode_counts','audit_lab.tests.test_batch1.FoundationalOracles.test_a_strict_parser_rejects_nonfinite_and_duplicate_keys'],
 'status':'حالت‌های مرزی آزمایشگاهی موفق؛ برابری تمام مسیرهای استثنای نادر با محیط Windows واقعی هنوز آزموده نشده است.'},
'E08':{
 'A':'برای متن، غایب/تهی/نوع غلط/سفید/متن معتبر را جدا می‌کند؛ دیگر متغیرها بیشتر دارای پرچم اعتبار هستند. صفر رأی مفید نگه داشته می‌شود.',
 'B':'`field_state` برای هر ستون، غایب/تهی/رشتهٔ خالی/آرایه یا دیکشنری خالی را تفکیک می‌کند؛ صفر معتبر باقی می‌ماند، اما در دو دسته.',
 'choice':'برای جلوگیری از اسکن اضافی ده‌ها میلیون سطر، جدول‌های کامل Q02 واقعی با ۱۰۲ ردیفِ فیلد-دسته-نقش استخراج شدند. این شاهد به‌عنوان Q، نه EDA، برچسب خورد. قرارداد A و صفرهای معتبر با آزمون کنترل شدند.',
 'tests':['test_e08_missing_nonblank_and_valid_zero_are_different','REAL_Q_FIELD_MISSINGNESS.csv'],
 'status':'بخشی: شاهد واقعی Q02 و آزمون A موجود است؛ نمایش کامل حالت‌های همهٔ فیلدها در گزارش جدید EDA همچنان نیازمند تصمیم و پیاده‌سازی است.'},
'E09':{
 'A':'حذف کپی دقیق جهانی فقط در هستهٔ معتبرِ توالی انجام می‌شد؛ توزیع پنج امتیاز پیش/پس به‌طور مستقل در خروجی قابل مشاهده نبود.',
 'B':'حساسیت خام/یکتای درون‌فایل را برای برخی مقادیر با SQLite و برابری نهایی رکورد پیاده‌سازی می‌کند، ولی تحلیل جهانی چهار دسته ندارد.',
 'choice':'محاسبهٔ پنج کلاس امتیاز در جامعهٔ core قبل از حذف و پس از حذف کپی دقیق سراسری به A اضافه شد. محاسبهٔ پس از حذف با تجمیع SQLite موجود انجام می‌شود، بدون بازخوانی یا ذخیرهٔ دوبارهٔ تمام نقدها. فاصلهٔ تغییر کل توزیع نیز ثبت می‌شود.',
 'tests':['test_e09_exact_canonical_duplicate_policy_A_and_B_agree_on_fixture','test_e09_global_core_star_sensitivity_matches_independent_row_oracle','REAL_Q_CATEGORY_COMPARISON.csv'],
 'status':'بخشی: حساسیت امتیاز هستهٔ چهار دسته روی دادهٔ مصنوعی پذیرفته شد؛ حساسیت تمام شاخص‌های تک‌متغیره و اعداد واقعیِ پس از حذف باید تکمیل شود.'},
'E10':{
 'A':'`Index.meta` از کلید محصول و برابری کانونی استفاده و فرادادهٔ چندنسخه‌ای را مبهم علامت‌گذاری می‌کند، بدون ضرب‌شدن ناخواستهٔ ردیف‌های نقد.',
 'B':'به‌صورت فایل‌مجزا تحلیل می‌کند و join چندنسخه‌ای محصول/نقد ندارد؛ بنابراین برای این الزام مبنای مناسب نیست.',
 'choice':'الگوریتم A نگه داشته شد. مورد جفت فرادادهٔ تکراری و دو نسخهٔ متعارض، محصول یکتا، یتیم و مخرج نقد بررسی شد.',
 'tests':['test_e10_unique_metadata_does_not_multiply_reviews','tests.test_eda.EndToEnd.test_join_does_not_multiply_or_trim_keys'],
 'status':'آزمون‌های join مصنوعی موفق؛ نرخ خطای اتصال روی فایل خام واقعی هنوز محاسبه نشده است.'},
'E11':{
 'A':'پیش از این تطبیق هش و شمارش با خروجی پوشهٔ کیفیت انجام می‌شد، اما برای ZIP واقعی کاربر قابل‌استفاده نبود و متریک‌ها کنترل مستقل نداشتند.',
 'B':'بخشی از خروجی کیفیت را فقط در قالب JSON مشخصی و با قرارداد هش مخصوص خود می‌خواند؛ قرارداد کامل Q01–Q25 و دامنهٔ چهار دسته را پیوند نمی‌دهد.',
 'choice':'خواندن بدون استخراج ZIP با تطبیق کدهای مرجع و دامنه، و ۱۳ کنترل تعریفاً یکسان در هر دسته (Q11 امتیازهای پنج‌گانه، Q13 زمان، Q14 قیمت و Q22 وجوه) پیاده شد. تناقض حتی یک شاخص یا یک فایل مانع استفادهٔ کل شواهد می‌شود.',
 'tests':['test_e11_equal_hashes_enable_10_Q_to_EDA_checkpoints','test_e11_scopes_and_mismatched_hashes_fail_closed','test_e03_and_e11_read_actual_Q_archive_without_reinterpreting_as_raw_scan','QUALITY_ARCHIVE_REJECTION'],
 'status':'اتصال و رد شواهد نامتناسب تأیید مصنوعی شد؛ Qهای واقعی به‌دلیل نبود هشت ورودی فعلاً فقط از نظر داخلی بررسی شده‌اند.'},
'E12':{
 'A':'پرچم‌های rating/time/text/image/metadata/price را در `RawCategory` با مخرج روشن تولید می‌کند؛ URL به معنای تصویر دانلودشده نیست.',
 'B':'عنوان، بدنه، تصاویر و فهرست‌های فراداده را با انواع داده بررسی می‌کند، ولی نمی‌تواند چهار دستهٔ اصلی را هم‌زمان پوشش دهد.',
 'choice':'تعریف محتوای معتبر A و Q بدون تغییر حفظ شد؛ ۱۳ کنترل نرخ دسترسی در هر دسته به گزارش Q قابل پیوند شد و واحد نقد در برابر شیء فراداده در آزمایش تثبیت شد.',
 'tests':['test_e12_e13_modality_rates_four_categories_have_local_denominators','REAL_Q_CATEGORY_COMPARISON.csv'],
 'status':'درون‌دادهٔ مصنوعی و شاهد واقعی Q موجود؛ بازاجرای EDA چهار دسته ضروری است.'},
'E13':{
 'A':'گزارش چهار‌دسته‌ای با مخرج‌های نقد/فراداده و برخی رابطه‌های شرطی دارد؛ تفکیک گمشدگی به‌عنوان سازوکار آماری اثبات نمی‌شود.',
 'B':'تنها دو دسته دارد و مقایسهٔ چهار دسته را به‌صورت واحد ارائه نمی‌کند.',
 'choice':'فراوانی‌های Q واقعی چهار دسته و ۱۰۲ حالت میدانی، همراه با جدول مقایسه با مخرج مشخص استخراج شدند. اختلاف درصدها صرفاً توصیفی تفسیر می‌شود؛ ادعای MCAR/MAR/MNAR یا آزمون علّی از آن استخراج نمی‌شود.',
 'tests':['test_e12_e13_modality_rates_four_categories_have_local_denominators','REAL_Q_CATEGORY_COMPARISON.csv','REAL_Q_FIELD_MISSINGNESS.csv'],
 'status':'مقایسهٔ توصیفی بر شواهد واقعی Q انجام شد؛ اثبات مکانیسم گمشدگی و هم‌جامعه‌بودن همهٔ گروه‌ها باز است.'},
'E31':{
 'A':'از شبکهٔ ثابت و کران‌های صدک استفاده می‌کند؛ نه صدک دقیق. برای مقدار ثابت، تهی و سرریز عددی وضعیت مشخص دارد.',
 'B':'روی شمارش کامل مقدارهای یکتا، صدک دقیق با قرارداد درون‌یابی خطی می‌گیرد؛ روی عددهای با کاردینالیتی بالا حافظهٔ متناسب با تعداد مقادیر متمایز نیاز دارد.',
 'choice':'A برای مقیاس بزرگ حفظ شد. کنترل بقای شمارش bin و وضعیت خطا افزوده شد؛ صدکِ nearest-rank مستقل باید داخل کران گزارش‌شده باشد. قرارداد متفاوت صدک B به‌اشتباه معادل A معرفی نشد.',
 'tests':['test_e31_nearest_rank_quantiles_are_inside_stated_bounds','test_e31_overflow_is_flagged_not_silently_reported_as_zero','test_e31_undefined_and_constant_stats_are_not_fake_results'],
 'status':'آزمون‌های عددی مرزی موفق؛ دقت تقریبی و عملکرد روی صدک‌های دادهٔ حجیم هنوز تأیید تجربی نشده است.'},
'E52':{
 'A':'پردازش ترتیبی JSONL و اندیس SQLite دارد؛ نگهداری تمام رکوردها در RAM انجام نمی‌شود، ولی هزینهٔ واقعی اندیس/دیسک برای ۷۰ میلیون نقد ثابت نشده است.',
 'B':'از Counter با کاردینالیتی بالقوهٔ زیاد برای صدک دقیق و جدول فراوانی استفاده می‌کند و اسکن کاراکتر به کاراکتر دارد؛ روی کل دادهٔ واقعی خطر RAM/زمان دارد.',
 'choice':'به مانیفست A، نمونه‌برداری حجم فایل SQLite و فضای آزاد پس از هر یک از هشت ورودی افزوده شد. یک آزمایش کوچک مستقل ۱۰هزار رکوردی اجرا شد؛ اعدادش فقط برای طراحی آزمایش واقعی استفاده می‌شوند، نه پیش‌بینی خطی زمان/دیسک.',
 'tests':['test_e52_sqlite_index_does_not_keep_10000_rows_in_python_list','test_e52_runtime_records_sqlite_disk_at_each_input_boundary','E52_MICRO_BENCHMARK.json'],
 'status':'صرفاً آزمایش فشار کوچک موفق؛ شرط مقیاس‌پذیری ۷۰ میلیون نقد هنوز تأیید نشده است.'},
}


def main():
    old=json.loads((ROOT/'audit_lab/evidence/REQUIREMENTS_TRIAGE_57.json').read_text(encoding='utf-8'))
    ledger=json.loads(json.dumps(old))
    statuses={}
    for r in ledger['requirements']:
        if r['id'] not in FINDINGS:
            continue
        item=FINDINGS[r['id']]
        r.update(audit_verdict='scoped_tests_passed_real_validation_pending',
                 acceptance_verified=False, scoped_synthetic_tests_passed=True,
                 evidence_ids=list(item['tests']), A_audit=item['A'], B_audit=item['B'],
                 selected_method=item['choice'], limitations=item['status'],
                 notes=item['status'])
        if r['id'] in ('E08','E09','E31','E52'):
            r['audit_verdict']='partially_implemented_or_validated'
        statuses[r['id']]=r['audit_verdict']
    ledger.update(schema_version=3, audit_round='BATCH2_FOUNDATIONS',
        batch2_19_tests_passed=True, legacy_tests_reconfirmed_separately={'E01':7,'A':16,'BATCH1':16,'BATCH2':19},
        raw_70m_EDA_executed_in_this_round=False, acceptance_verified_counts_all_real=0,
        batch2_verdicts=statuses,source_quality_comparison='batch2_real_q/REAL_Q_CATEGORY_COMPARISON.csv')
    output=ROOT/'audit_lab/evidence/batch2'
    output.mkdir(parents=True,exist_ok=True)
    (output/'REQUIREMENTS_LEDGER_57_AFTER_BATCH2.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'VERDICTS_E03_E13_E31_E52.json').write_text(json.dumps({
        'source':'da_p.pdf pp.6-7; slides03.pdf; PROJECT_CARD.md; archived Q01-Q25; candidate A/B',
        'count':len(FINDINGS),'decisions':FINDINGS,'all_real_data_accepted':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    real=(ROOT/'audit_lab/evidence/batch2_real_q/REAL_Q_CATEGORY_COMPARISON_FA.md').read_text(encoding='utf-8')
    bench=json.loads((output/'E52_MICRO_BENCHMARK.json').read_text())
    regression=json.loads((output/'REGRESSION_B2_VS_E02.json').read_text())
    lines=['# ممیزی گروهی بنیادین — ادامهٔ E02: E03 تا E13 و E31 و E52','',
        '## دامنه و روش','',
        'مرجع کد: بستهٔ تجمیعی E02؛ E01 و E02 حفظ شده‌اند. برای هر الزام مستقل اصل/منبع، بررسی A، بررسی B، روش منتخب/اصلاح و دلیل پذیرش یا بازماندن ثبت شده است.',
        'سند رسمی پروژه `da_p.pdf` صص ۶–۷؛ `slides03.pdf` صص ۳–۱۱ و ۱۵–۱۸؛ `slide02.pdf` صص ۱۸–۲۰ و `PROJECT_CARD.md`، منابع تصمیم‌اند.',
        'نتایج واقعی فقط از بستهٔ Q01–Q25 از پیش اجراشده استخراج شده‌اند؛ هشت فایل JSONL اصلی در این محیط نیستند.', '',
        '## تغییرات تولیدی','',
        '- `eda/quality_link.py` جدید: پیوند ZIP یا پوشهٔ کیفیت فقط با تطبیق چهار ورودی، نسخه، هش، scope، timestamp و ۱۳ شاخص متناظر در هر دسته؛ در تناقض، امتناع از استناد.',
        '- `eda/stats.py`: رد شمارش ناممکن و کنترل بقای توزیع.',
        '- `eda/univariate.py`, `eda/index.py`, `eda/products.py`: حساسیت توزیع پنج ستارهٔ جامعهٔ معتبر core قبل/بعد از حذف کپی دقیق در چهار فایل، بدون اسکن دوباره.',
        '- `eda/__main__.py`: گزینهٔ `--quality-archive` و پایش دیسک/حجم SQLite در مرز ورودی‌ها.',
        '- کدهای تثبیت‌شدهٔ `quality_audit.py` و `structure_audit.py` و قرارداد E01 تغییر نکرده‌اند.','',
        '## آزمایش‌ها و بازگشت','',
        '۴ مجموعه به‌طور مجزا اجرا شدند: E01 = ۷ موفق، A = ۱۶ موفق، E02 = ۱۶ موفق، این نوبت = ۱۹ موفق؛ جمع ۵۸ آزمون موفق در اجراهای مستقل.',
        f"کنترل خروجی هم‌پارامتر با E02: {regression['identical_count']} فایل بی‌تغییر، {regression['changed_count']} فایل با اختلاف مقصود، همهٔ {regression['figures_unchanged']} نمودار عیناً بی‌تغییر.",
        'تلاش اجرای یک‌جای همهٔ تست‌ها به timeout برخورد کرد و به‌عنوان آزمون موفق شمارش نشده است؛ لاگ مستقل برای هر مجموعه محفوظ است.','',
        '## ماتریس وضعیت','',
        '| کد | نتیجهٔ فعلی | آزمون/شاهد |','|---|---|---|']
    for r in ledger['requirements']:
        if r['id'] in FINDINGS:
            item=FINDINGS[r['id']]
            lines.append(f"| {r['id']} | {item['status']} | {', '.join(item['tests'][:2])} |")
    lines.extend(['','## گزارش مستقل پنج‌بخشی هر الزام',''])
    for r in ledger['requirements']:
        if r['id'] not in FINDINGS: continue
        item=FINDINGS[r['id']]
        lines.extend([f"### {r['id']} — {r['title']}", '',
            f"**۱) اصل الزام / منبع و معیار پذیرش.** {r['definition']} منبع: {r['source']}، ماهیت: {r['classification']}. **معیار آزمون:** {r['acceptance']}", '',
            f"**۲) بررسی A.** {item['A']}", '',
            f"**۳) بررسی B و مقایسه.** {item['B']}", '',
            f"**۴) روش منتخب و اصلاح.** {item['choice']}", '',
            f"**۵) اثبات / موارد مرزی / وضعیت.** آزمون‌ها: `{', '.join(item['tests'])}`. **حکم فعلی:** {item['status']}", ''])
    lines.extend(['','## مقایسهٔ واقعیِ قابل استناد از Q01–Q25','',real,'',
        '## آزمون مقیاس‌پذیری کوچک E52','',
        f"اندیس SQLite روی {bench['records_requested']:,} ردیف ساختگی: {bench['index_size_bytes_at_boundary']:,} بایت فایل دیتابیس، ",
        f"{bench['tracemalloc_peak_traced_bytes']:,} بایت حافظهٔ peak ردیابی‌شدهٔ پایتون (نه کل RSS)، "
        f"{bench['elapsed_seconds']:.3f} ثانیهٔ پردازش در این محیط. این عددها به ۷۰ میلیون رکورد قابل تعمیم نیستند.", '',
        '## موارد باز و نوبت بعد','',
        'E08 برای نمایش همهٔ حالت‌های فیلدی در EDA، E09 برای حساسیت دیگر متغیرها و E52 برای اجرای پرحجم واقعی همچنان نیمه‌تمام‌اند؛ هیچ‌یک به‌عنوان قبولی علمی کامل درج نشده‌اند.',
        'نوبت بعد مطابق برنامهٔ مصوب، ممیزی تک‌متغیرهٔ E14 تا E30 خواهد بود؛ اما آزمون‌های این نوبت در تمام مراحل بعد حفظ می‌شوند.',
        'اجرای واقعی چهار دسته و تطبیق دوبارهٔ هش فایل‌های خام باید پیش از تأیید نهایی انجام شود.'])
    (output/'BATCH2_AUDIT_REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'CHANGELOG_E03_E13_E31_E52_FA.md').write_text('''# تغییرات نسبت به بستهٔ E02

- جدید: `eda/quality_link.py`، خواندن مستقیم ZIP شواهد Q01–Q25 و ۱۳ کنترل در هر دسته با رد اختلاف.
- اصلاح: قرارداد دقیق `frequency`، کنترل شمارش توزیع، حساسیت توزیع core پس از حذف دقیق سراسری.
- جدید: پایش دیسک SQLite در هشت مرز ورودی و بنچمارک کوچک مستقل.
- جدید: خروجی واقعی مقایسهٔ چهار دسته و ۱۰۲ ردیف حالت گمشدگی از بستهٔ کیفیت مرجع.
- جدید: ۱۹ آزمون، دفتر مستقل ۵۷ الزام، گزارش پنج‌بخشی و شواهد کنترل بازگشت.
- ثابت: منطق Q01–Q25، سورس ساختار، نسخهٔ قرارداد E01، ساختار رسمی دسته‌ها.
- تأیید نشده: اجرای کامل خام ۷۰٬۶۲۸٬۴۶۷ نقد و آزمون مقیاس‌پذیری در مقیاس حقیقی.
''',encoding='utf-8')
    print('Written audit report and 57-row audit ledger')

if __name__=='__main__':main()
