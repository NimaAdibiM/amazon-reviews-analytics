"""Generate a separate five-section verdict for each E14..E30 without false passes."""
from pathlib import Path
import csv,json
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_lab/evidence/batch3'
# Each entry: A, B, fault, selection, tests, status, remaining
F={
14:('شمارندهٔ پنج ستاره و مخرج امتیاز معتبر در RawCategory موجود بود.','شمارنده و شاخص rating_balance در B موجود بود.',
    'نتیجهٔ خامِ کلاس‌ها در هر دو روش قابل‌مقایسه است؛ برخی گزارش‌ها قبلاً فقط سهم ۱–۲ را برجسته می‌کردند.',
    'جدول مستقل پنج رده با رده‌های صفر و بقای مجموع در JSON/Markdown افزوده شد.','test_e14_all_star_classes_and_invalid_ratings','synthetic_pass_pending_real','برابرشدن شمارهای واقعی با Q11 در اجرای کامل EDA باید اثبات شود.'),
15:('A سهم کم‌ستاره را ثبت می‌کرد و جامعهٔ ۵+۱۰ را از آن جدا نگه می‌داشت.','B سهم‌های کم/خنثی/زیاد، رده‌های غایب، نسبت کلاس و آنتروپی را داشت.',
    'خلاصهٔ اولیهٔ A همهٔ نماهای عدم‌توازن را یک‌جا منتشر نمی‌کرد.',
    'کلاس‌های ۱–۲/۳/۴–۵، نسبت پنج کلاس، فقدان کلاس‌ها و آنتروپی نرمال‌شده با حفظ مخرج valid ratings افزوده شد.','test_e15_ordinal_imbalance_and_empty_classes','synthetic_pass_pending_real','عدم‌توازن نقد خام معادل توزیع هدف ده نقد آینده نیست.'),
16:('A طول بدنه/عنوان را فقط برای متن ناتهی می‌گرفت.','B تمام مقادیر رشته‌ای حتی رشتهٔ سفید/خالی را اندازه می‌گرفت؛ طول عنوان فقط نویسه بود.',
    'این دو مخرج با هم قابل‌قیاس نبودند و Q22 بدنه‌های سفید را در سنجهٔ طول رشته‌ای وارد می‌کند.',
    'نمای قدیمی nonblank حفظ و چهار سنجهٔ all-string نویسه/کلمه برای عنوان/بدنه افزوده شد.','test_e16_blank_body_string_zero_not_missing;test_independent_B_implementation_agrees_on_shared_fixture_metrics','synthetic_pass_pending_real','هم‌ارزی واقعی همهٔ سنجه‌های رشته‌ای با Q22 فقط پس از انطباق هشت هش ثابت می‌شود.'),
17:('A دو پرچم اصلی Q23 را با مخرج nonblank گزارش می‌کرد.','B آستانهٔ پیش‌فرض کوتاه تا ۵ واژه و بلند ۵۰۰ واژه داشت؛ تعریف متفاوتی است.',
    'مخرج «بلند» باید تمام بدنه‌های رشته‌ای باشد، نه لزوماً بدنهٔ ناتهی.',
    'دو نرخ مستقل با آستانهٔ رسمی ۳ واژه و ۱۰هزار code point و مخرج‌های منطبق Q23 به خروجی افزوده شد.','test_e17_q23_boundary_and_denominators;test_Q22_Q23_crosswalk_rejects_a_mismatched_all_string_count','synthetic_pass_pending_real','آستانه‌ها توصیفی و از تشخیص کیفیت معنایی متن جدا هستند.'),
18:('A پرچم non-ASCII، replacement و surrogate ثبت می‌کرد.','B پروفایل اسکریپت و fastText با شرط confidence دارد، اما مدل خارجی نصب‌نشده و فقط دو دسته را می‌پوشاند.',
    'اسکریپت زبان نیست؛ برچسب زبان با مدل اجرا‌نشده قابل تأیید نیست.',
    'در گزارش A وضعیت language=not_computed ثبت شد؛ Q24 آرشیوی فقط شاهد ثانوی با دامنهٔ مستقل است.','test_e18_script_is_not_a_language_label;test_real_Q22_Q23_Q24_are_archived_evidence_not_a_new_EDA_scan','partial_unverified_language','مدل زبانی نسخه‌دار، آزمون خطا روی متن‌های Amazon، اجرای واقعی و تضمین استفادهٔ صحیح از آستانه‌ها باز است.'),
19:('A فراوانی صفر، میانگین/SD و طبقات توان‌دویی رأی را ثبت می‌کرد.','B صدک‌های دقیق، چولگی/کشیدگی و نمودار ECDF دارد ولی Counter برای همهٔ مقادیر یکتا نگه می‌دارد.',
    'آمار شکل توزیع و دنباله در A کم بود و B روی دامنهٔ بزرگ حافظهٔ ثابت تضمین نمی‌کند.',
    'چولگی/کشیدگی از لحظه‌های آنلاین، صفر واقعی، بیشینه و بازهٔ محافظه‌کارانهٔ دنباله به A افزوده شد.','test_e19_votes_zero_retained_and_invalid_removed;test_e27_streaming_shape_matches_independent_moments','synthetic_pass_pending_real','نسخهٔ کامل هنوز نیازمند سنجش پایداری لحظه‌های مرتبهٔ بالا در مقیاس واقعی است.'),
20:('A سه حالت true/false/unusable را بدون تبدیل ضمنی ثبت می‌کرد.','B true/false را در دسته‌ای و null/wrong را در field states نگه می‌داشت.',
    'خطر تفسیر false به نقد جعلی و تفسیر ۱ عددی به true بود.',
    'قرارداد A حفظ و سناریوهای bool/string/int/null آزموده شد؛ هشدار تفسیر در گزارش جدید ثبت شد.','test_e20_purchase_flags_neither_coerce_nor_infer_fraud','synthetic_pass_pending_real','نرخ «خرید تأییدشده» نشانهٔ صحت یا کیفیت نقد نیست.'),
21:('A سال UTC، بازهٔ زمان و شمار زمان معتبر را ثبت می‌کرد.','B سال و ماه UTC و ماه‌های بدون رکورد را داشت.',
    'نمای ماهانهٔ A وجود نداشت؛ سکوت یک ماه ممکن بود با نرخ صفر بازار اشتباه شود.',
    'شمار دقیق ماه‌ها در همان اسکن، پرکردن صفر ماه‌های داخلی و خروجی نمودار/CSV با هشدار دورهٔ مرزی افزوده شد.','test_e21_utc_gap_months_and_partial_boundary','synthetic_pass_pending_real','مواجهه و پوشش ماهانه ثابت نیست؛ روند خرید یا تقاضا ادعا نمی‌شود.'),
22:('A قیمت، میانگین ستاره و شمار ستاره را معتبرسازی و توزیع snapshot می‌کرد.','B توصیف دقیق‌تری از قیمت و دیگر فراداده در واحد metadata record ارائه می‌کرد.',
    'میانگین نمایش‌داده‌شده از رتبهٔ تاریخی پنجم قابل بازیابی نیست.',
    'قرارداد A و صفر قیمت حفظ شد؛ نمودار اختصاصی میانگین امتیاز با bin=0.25، گزارش زمان خزش افزوده شد.','test_e22_metadata_snapshot_zero_price_and_invalid','synthetic_pass_pending_real','تحلیل محصولی همگانی از فرادادهٔ موجود به کل محصولات تعمیم داده نمی‌شود.'),
23:('A دستهٔ فرادادهٔ main_category/store را دقیق نمی‌شمرد.','B فراوانی هر سطح را در Counter و نمودار top12 داشت؛ برای تعداد سطح بالا رشد حافظه نامحدود است.',
    'کمبود فراوانی اسمی و خطر رشد شدید واژگان دسته/فروشگاه در A.',
    'تعداد و top12 برچسب خام exact را در SQLite روی دیسک جمع می‌کنیم؛ Other تعداد سطوح و رکوردها دارد.','test_e23_e28_exact_nominals_all_records_including_missing_parent','partial_nested_categories','فهرست‌های درونی categories فعلاً فقط طول دارند؛ بررسی توزیع برچسب‌های چندسطحی باقی است.'),
24:('A حضور URL/متن قابل‌استفاده را به‌درستی از تعداد محتوا جدا می‌کرد.','B تعداد ورودی‌های features/description/images/videos و چند طول محتوا را داشت.',
    'A توزیع طول فهرست‌ها و تعداد ورودی‌ها را نداشت؛ تعداد URL چند رزولوشن نباید تعداد تصویر فرض شود.',
    'سنجه‌های ساختاری B در A به‌صورت streaming، با صفر فهرست خالی و رد نوع اشتباه پیاده شد.','test_e24_list_entry_count_preserves_empty_lists','synthetic_pass_pending_real','وجود URL اتصال تصویر واقعی یا موفقیت دانلود نیست.'),
25:('A تعداد نقد بر حسب parent را با SQLite محاسبه می‌کرد؛ توزیع user نداشت.','B Counter جداگانهٔ reviewer و parent با نما و تمرکز داشت، اما حافظه تابع تعداد شناسه است.',
    'فقدان توزیع کاربر در A و نیاز مقیاس‌پذیری برای میلیون‌ها شناسه.',
    'تجمیع دقیق user در SQLite، گروه‌های حجم و سهم ۱٪ پرتکرار افزوده شد؛ هیچ ID منتشر نمی‌شود.','test_e25_user_activity_includes_invalid_parent_and_no_user_export;test_e26_weighted_grouped_moments_match_expanded_oracle','synthetic_pass_pending_real','حساب جهانی کاربر در چهار دسته و ارزیابی هزینهٔ واقعی SQLite هنوز انجام نشده است.'),
26:('A لحظه‌های population و کران‌های صدک log2 داشت.','B صدک دقیق با جداول فراوانی کامل، نما و MAD داشت.',
    'کران صدک A با صدک نقطه‌ای B یکسان نیست و خطای گزارش‌نکردن این تفاوت جدی است.',
    'کران صدک با روش nearest-rank، بررسی بقا، momentهای آنلاین و ستون‌های شفاف خروجی حفظ/تقویت شد؛ محاسبهٔ exact برای همهٔ متغیرها اجباری نشده است.','test_e26_nearest_rank_brackets_contain_reference_quantiles;test_e26_weighted_grouped_moments_match_expanded_oracle','partial_exact_quantiles','نمای دقیق و چارک دقیق برای مقادیر پرکاردینال هنوز نداریم؛ باید تصمیم علمی/محاسباتی گرفته شود.'),
27:('A min/max و صدکِ بازه‌ای داشت اما شکل لحظه‌ای و دنبالهٔ quantile کمتر بود.','B skew/kurtosis، Gini و دنباله را exact از Counter گزارش می‌کرد.',
    'نمای B بر دادهٔ حجیم خطر حافظه دارد؛ در A شکل و مرز ناممکن/پرت برای همهٔ شاخص‌ها کامل نبود.',
    'ممان‌های سوم/چهارم آنلاین، چولگی/کشیدگی و شمار محافظه‌کارانهٔ بالای P95 افزوده شد. عدم‌محاسبه در حالت ثابت/سرریز صریح است.','test_e27_streaming_shape_matches_independent_moments;test_e19_votes_zero_retained_and_invalid_removed','partial_outlier_precision','تعداد پرتِ دقیق با قاعدهٔ مشخص و یکپارچه‌سازی Q15–Q18 در همهٔ شاخص‌ها باز است.'),
28:('A بخش اسمی را برای فراداده کمتر پوشش می‌داد.','B نما، سهم سطح غالب، سایر سطوح و آنتروپی اسمی داشت.',
    'دسته‌بندی raw store/main_category به شناسهٔ عددی جعلی یا میانگین اسمی نباید منجر شود.',
    'top-k exact با مخرج nonblank، تعداد سطح distinct و Other در SQLite ثبت شد؛ برای ستارهٔ ترتیبی پنج رده حفظ شد.','test_e23_e28_exact_nominals_all_records_including_missing_parent','partial_entropy_nested','آنتروپی همهٔ سطوح و زیرمجموعه‌های categories هنوز برای همهٔ ورودی‌ها محاسبه نشده است.'),
29:('A توزیع امتیاز، متن، رأی، محصول، زمان، قیمت را در ۱۳ نمودار اولیه تولید می‌کرد.','B هیستوگرام+ECDF+box برای هر متغیر و نمودارهای متعدد داشت.',
    'یک چارت قدیمی همهٔ ساختارهای فراداده/کاربر/ماهانه را پوشش نمی‌داد.',
    'یک نمودار تک‌متغیرهٔ تکمیلی ۹بخشی برای هر دسته اضافه شد؛ ۱۳ تصویر قبلی دست‌نخورده ماند.','test_e29_e30_output_units_months_diagrams_and_denominators;REGRESSION_COMPARISON_B2_TO_B3','synthetic_pass_pending_real','نمایش چهار دستهٔ واقعی و بررسی خوانایی برچسب‌های پرتعداد باید روی خروجی واقعی انجام شود.'),
30:('A عنوان، n و برچسب نوع فراوانی دارد؛ quantile binها گزارش می‌شوند.','B نمودارهای تمام‌دامنهٔ جدا و اعلام دنبالهٔ خارج از نمایش مرکزی دارد.',
    'امکان اشتباه گرفتن نقاط برش نمایشی و دادهٔ حذف‌شده یا mخرج‌های متفاوت وجود داشت.',
    'نمودارهای جدید صریحاً raw/valid، مقیاس log2، اسامی برش‌خورده و خروجی کامل CSV را مشخص می‌کنند؛ هیچ داده‌ای حذف نمی‌شود.','test_e29_e30_output_units_months_diagrams_and_denominators;REGRESSION_COMPARISON_B2_TO_B3','partial_visual_review','پالایش نهایی تک‌تک برچسب‌ها و نمودارهای ۸ ورودی کامل هنوز باز است.'),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    req={r['id']:r for r in csv.DictReader((ROOT/'EDA_PHASE1_REQUIREMENTS_E01_E57_FA.csv').open(encoding='utf-8-sig'))}
    assert len(F)==17
    results={}
    lines=['# ممیزی پنج‌بخشی تک‌متغیره: E14 تا E30','',
           '**وضعیت:** اصلاح عملی روی بستهٔ تجمیعی نوبت دوم؛ بدون تأیید نهایی اجرای خام واقعی.','',
           '## مبنای شواهد', '',
           '- سند رسمی پروژه `da_p.pdf`، صفحات ۶ و ۷؛ اسلایدهای `slides03.pdf` صفحات ۳–۹ و ۱۷؛ `slide02.pdf` صفحات ۱۳–۲۰؛ نوت‌بوک درس، گام ۷ (سلول‌های ۹۵–۱۰۸).',
           '- کارت پروژه و قرارداد E01؛ کدهای A و B؛ نتایج آرشیوی Q11 و Q22–Q24 از چهار دسته.',
           '- اجرای مستقل B بر همان fixture، آزمون‌های آماری مستقل و کنترل بازگشت A.',
           '- صورت و مخرج جداگانه؛ جامعهٔ خام نقد، شیء فراداده، محصول و کاربر جابه‌جا نمی‌شوند.', '',
           '## جدول نتیجهٔ مستقل', '',
           '| الزام | نتیجهٔ مصنوعی | تأیید واقعی کامل | توضیح |','|---|---|---|---|']
    for n in sorted(F):
        a,b,fault,choice,tests,status,remaining=F[n];id=f'E{n:02d}'
        results[id]=dict(title=req[id]['title'],source=req[id]['source'],nature=req[id]['nature'],
                         acceptance=req[id]['acceptance'],candidate_A=a,candidate_B=b,
                         confirmed_problem=fault,selected_implementation=choice,
                         test_ids=tests.split(';'),verdict=status,real_full_data_verified=False,
                         remaining=remaining)
        lines.append(f'| {id} | آزمون محدود موفق | خیر | {remaining} |')
    lines+=['','## ممیزی الزام‌به‌الزام (پنج بخش مستقل)','']
    for n in sorted(F):
        id=f'E{n:02d}';v=results[id]
        lines+= [f'### {id} — {v["title"]}','',
                 f'**۱. اصل الزام، منبع و معیار پذیرش:** {req[id]["definition"]}؛ منبع: {v["source"]}؛ نوع: {v["nature"]}. معیار: {v["acceptance"]}.','',
                 f'**۲. بررسی A:** {v["candidate_A"]}','',
                 f'**۳. بررسی B و مقایسه:** {v["candidate_B"]} ایراد/خطر یا اختلاف اثبات‌شده: {v["confirmed_problem"]}','',
                 f'**۴. روش منتخب و اصلاح واقعی:** {v["selected_implementation"]}','',
                 f'**۵. اثبات درستی و وضعیت:** آزمون‌ها: `{", ".join(v["test_ids"])}`. '
                 f'وضعیت: `{v["verdict"]}`؛ **تأیید کامل واقعی: خیر**. باز: {v["remaining"]}','']
    lines+=['','## محافظ‌های علمی مهم','',
            '۱. آمار همهٔ رشته‌ها و متن ناتهی با مخرج متفاوت‌اند؛ نتایج Q22/Q23 فقط در تعریف یکسان قابل مقایسه است.',
            '۲. برچسب زبان با اسکریپت جایگزین نمی‌شود؛ بدون مدل نسخه‌دار، E18 باز است.',
            '۳. قدرت B در صدک/جداول فراوانی دقیق به قیمت رشد حافظه با تعداد مقدار یکتا است. برای A کران واقعی صدک گزارش شده، نه عدد جعلی.',
            '۴. شاخص‌هایی مثل skewness/kurtosis جمعیت بر مبنای n هستند؛ این قرارداد با `pandas.std()` پیش‌فرض که `ddof=1` دارد متفاوت است.',
            '۵. پس از اصلاح، Q01–Q25 و کارت پروژه تغییری نکرده‌اند؛ مرجع خام واقعی در این محیط وجود ندارد.','',
            '## اجرای مجدد و شواهد','',
            '`python -m unittest audit_lab.tests.test_univariate_batch3 -v`',
            'اجرای کامل در محیط مصنوعی: `python -m tests.fixture NEW_INPUT_DIR` و `python -m eda --config ... --mode synthetic --out NEW_OUTPUT_DIR`.',
            'قبل از اجرای واقعی، همسانی نام/نقش و هش هشت فایل باید تأیید شود. هیچ شاخصی صرفاً به‌دلیل وجود کد قبول نهایی نیست.','']
    OUT.joinpath('VERDICTS_E14_E30.json').write_text(json.dumps({'count':len(results),'decisions':results,'formal_full_real_passes':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    OUT.joinpath('BATCH3_AUDIT_REPORT_FA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    old=list(csv.DictReader((ROOT/'audit_lab/evidence/batch2/REQUIREMENTS_LEDGER_57_AFTER_BATCH2.csv').open(encoding='utf-8-sig')))
    assert len(old)==57
    for row in old:
        id=row['id']
        if id in results:
            v=results[id]
            row['audit_verdict']=v['verdict']
            row['acceptance_verified']='False'
            row['scoped_synthetic_tests_passed']='True'
            row['notes']=v['remaining']
            row['evidence_ids']=','.join(v['test_ids'])
    ledger=OUT/'REQUIREMENTS_LEDGER_57_AFTER_BATCH3.csv'
    with ledger.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)
    OUT.joinpath('CHANGELOG_E14_E30_FA.md').write_text('''# تغییرات نوبت سوم نسبت به نوبت دوم

- حفظ کامل E01 و E02 و منطق مرجع کیفیت؛ هیچ تغییر به `quality_audit.py` یا `structure_audit.py` داده نشده است.
- گسترش `eda/univariate.py` برای E14–E24، متریک‌های رشته‌ای شامل متن خالی، ماه‌های UTC، کلاس‌های پنج‌گانه و ساختار فراداده.
- افزودن `eda/index.py` برای شمارش exact روی دیسک نام فروشگاه/دسته و کاربران؛ ID خام هرگز صادر نمی‌شود.
- افزودن `eda/stats.py` برای لحظه‌های مرتبهٔ سوم/چهارم با حافظهٔ ثابت و شمار محافظه‌کارانهٔ دنبالهٔ P95.
- افزودن چهار نمودار `univariate_extended.png` و فایل‌های JSON/CSV/Markdown در خروجی جدید.
- افزودن چهار کنترل اختیاری و سخت‌گیرانهٔ Q22/Q23 در صورت وجود شواهد واقعی سازگار؛ حالت تست مصنوعی بدون Q22/Q23 تغییری نکرد.
- افزودن ۱۹ آزمون جدید و اجرای واقعی candidate B روی fixture یکسان؛ افزایش تعداد نمودارها در آزمون‌های قدیمی با دلیل اصلاح شد.
- کیفیت مرجع و تمام خروجی‌ها در حالت synthetic از خروجی واقعی ۷۰میلیونی جدا نگه داشته شدند.
''',encoding='utf8')
    print('Created',len(results),'five-section reports and',len(old),'ledger records.')

if __name__=='__main__':main()
