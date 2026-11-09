"""E01: validate the immutable research question contract against the project card.

No data mutation, model training, feature fitting, or dataset IO occurs here.
"""
import hashlib
import json
from pathlib import Path

CONTRACT_FILENAME = 'RESEARCH_CONTRACT_E01.json'
REQUIRED_CATEGORIES = ('All_Beauty', 'Appliances', 'Beauty_and_Personal_Care', 'Electronics')
EXPECTED_REVIEWS = (tuple(range(1, 6)), tuple(range(6, 16)))
EXPECTED_STARS = (1, 2)


def load_research_contract(project_root):
    project_root = Path(project_root)
    path = project_root / CONTRACT_FILENAME
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('E01: research contract must be a JSON object')
    if data.get('requirement') != 'E01' or data.get('contract_version') != '1.0.0':
        raise ValueError('E01: unrecognized research contract version')
    if data.get('source_document') != 'PROJECT_CARD.md' or data.get('source_document_version') != '1.1':
        raise ValueError('E01: unexpected source project card')
    card_path = project_root / 'PROJECT_CARD.md'
    card_bytes = card_path.read_bytes()
    card_hash = hashlib.sha256(card_bytes).hexdigest()
    if data.get('source_document_sha256') != card_hash:
        raise ValueError('E01: PROJECT_CARD.md changed; review and explicitly reapprove research contract')
    card = card_bytes.decode('utf-8')
    # Ground the exact question in the named section, not in an unrelated occurrence.
    try:
        section = card.split('## 6. Main research question and hypotheses', 1)[1].split('## 7. Target variable', 1)[0]
        quoted_question = next(line[2:] for line in section.splitlines() if line.startswith('> '))
    except (IndexError, StopIteration) as e:
        raise ValueError('E01: primary research question missing from project card') from e
    if data.get('research_question_verbatim') != quoted_question:
        raise ValueError('E01: research question does not match project card')
    if sorted(data.get('categories', [])) != list(REQUIRED_CATEGORIES):
        raise ValueError('E01: categories do not match four-category research design')
    time = data.get('time', {})
    if tuple(time.get('input_review_indices', [])) != EXPECTED_REVIEWS[0] or tuple(time.get('outcome_review_indices', [])) != EXPECTED_REVIEWS[1]:
        raise ValueError('E01: first-five/next-ten indexing has drifted')
    target = data.get('target', {})
    if target.get('denominator') != 10 or target.get('low_star_labels') != list(EXPECTED_STARS) or target.get('three_star_is_low') is not False:
        raise ValueError('E01: low-star target definition has drifted')
    if target.get('support') != [round(i / 10, 1) for i in range(11)]:
        raise ValueError('E01: target support must be 0, 0.1, ..., 1')
    if data.get('primary_hypothesis', {}).get('comparison') != 'S versus ST':
        raise ValueError('E01: primary comparison must isolate added text')
    if data.get('study_unit', {}).get('product_key') != 'parent_asin':
        raise ValueError('E01: predictive unit must be globally identified parent product')
    if data.get('phase1_boundary', {}).get('do_not_train_or_evaluate_models') is not True:
        raise ValueError('E01: Phase 1 EDA must not claim validated model gains')
    if 'Y_p = (z_p6 + ... + z_p15) / 10' not in card:
        raise ValueError('E01: target formula missing from project card')
    return data


def research_markdown(data, scope=None):
    lines = [
        '# E01 — قرارداد سؤال پژوهشی و حدود تحلیل اکتشافی', '',
        f'**منبع:** `PROJECT_CARD.md` نسخهٔ {data["source_document_version"]}؛ '
        f'هش SHA-256: `{data["source_document_sha256"]}`.', '',
        '## سؤال اصلی', '',
        '> ' + data['research_question_verbatim'], '',
        '**خلاصهٔ فارسی:** ' + data['question_summary_fa'], '',
        '## واحد، زمان و هدف', '',
        '- واحد پیش‌بینی: هر محصول با شناسهٔ جهانی `parent_asin`؛ نه هر نقد.',
        '- زمان تصمیم: پس از نقد پنجم معتبر و غیرتکراری مشاهده‌شده (`t5`).',
        '- ورودی: امتیاز، زمان و در حالت ST متن نقدهای ۱ تا ۵؛ برچسب از نقدهای ۶ تا ۱۵.',
        '- افق: **ده نقد مشاهده‌شدهٔ بعدی**؛ نه ده روز یا افق تقویمی ثابت.',
        '- هدف: سهم امتیازهای ۱ و ۲ در ده نقد بعدی، با مقادیر ممکن ۰ تا ۱ در گام‌های ۰٫۱.',
        '- محصول بدون ده نقد آینده، برچسب نامشخص دارد؛ برچسب صفر داده نمی‌شود.', '',
        '## مقایسهٔ علمی و تصمیم', '',
        '- فرضیهٔ اصلی: مقایسهٔ منصفانهٔ S با ST برای **ارزش افزودهٔ متن**، با محصولات، تقسیم و بودجهٔ تنظیم یکسان.',
        '- خط‌های مبنای ثانویه B0 و B1 جایگزین مقایسهٔ اصلی S/ST نیستند.',
        '- خروجی عملیاتی مدنظر: رتبه‌بندی محصولات واجدشرایط برای بازبینی انسانی؛ بودجهٔ ۱۰٪ فرض طراحی است، نه ظرفیت اندازه‌گیری‌شده.', '',
        '## وضعیت فاز اول و محدودیت ادعا', '',
        '- این مرحله صرفاً داده‌ها، روابط، جامعهٔ قابل‌ارزیابی و خطرهای نشت اطلاعات را توصیف می‌کند.',
        '- هیچ بهبود پیش‌بینی خارج از نمونه، اثر علّی یا منفعت اقتصادی در این گزارش اثبات نشده است.',
        '- نمایش توصیفی برچسب آینده به معنی مجازبودن ورود آن به ویژگی‌های پیش‌بینی نیست.',
        '- مرزهای قطعی تقسیم زمانی و زمان مجاز برچسب‌ها هنوز نهایی نشده‌اند.',
        '- تفاوت دادهٔ خام با جمعیت ۵+۱۰ و سوگیری شرط داشتن ۱۵ نقد باید جدا گزارش شود.',
    ]
    if scope is not None:
        lines.extend(['', f'**دامنهٔ این اجرا:** `{scope}`؛ این برچسب، نتیجهٔ پیش‌بینی واقعی را ثابت نمی‌کند.'])
    return '\n'.join(lines) + '\n'
