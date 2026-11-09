"""Stable CSV/JSON exports and Persian analysis reports."""
import csv
import json


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def rate_rows(value, path=()):
    """Export only objects carrying a numerator, denominator and fraction."""
    if isinstance(value, dict):
        if {"numerator", "denominator", "fraction"} <= value.keys():
            context = {k: v for k, v in value.items() if k not in ("numerator", "denominator", "fraction")}
            yield (".".join(path), value["numerator"], value["denominator"], value["fraction"],
                   json.dumps(context, ensure_ascii=False, separators=(",", ":")))
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from rate_rows(child, (*path, key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from rate_rows(child, (*path, str(i)))


def rate_csv(path, payload):
    write_csv(path, ("json_path", "numerator", "denominator", "fraction", "context_json"), rate_rows(payload))


def distribution_csv(path, distributions):
    cols = ("measure", "measurement_unit", "total", "valid", "excluded", "min", "max",
            "mean", "population_sd", "p25_bounds", "p50_bounds", "p75_bounds", "p95_bounds",
            "iqr_bounds", "zero_n", "bin_rule", "moment_status")
    write_csv(path, cols,
              ((key, d.get("measurement_unit"), d["total"], d["valid"], d["excluded"],
                d["minimum"], d["maximum"], d["mean"], d["population_sd"],
                *(d["quantile_bounds"].get(q) for q in ("p25", "p50", "p75", "p95")),
                d["iqr_bounds"], d["zero"]["numerator"], d["bin_rule"], d["moment_status"])
               for key, d in distributions.items()))


def value(x):
    if x is None:
        return "تعریف‌نشده"
    if isinstance(x, float):
        return f"{x:,.5g}"
    return str(x)


def percent(row):
    n, d, p = row["numerator"], row["denominator"], row["fraction"]
    return f"{n:,} از {d:,}" + (f" ({p:.1%})" if p is not None else " (نسبت تعریف‌نشده)")


def distribution_text(d):
    q = d["quantile_bounds"]
    if not d["valid"]:
        return f"معتبر ۰، خارج از این خلاصه {d['excluded']:,}."
    return (f"معتبر {d['valid']:,}، خارج از این خلاصه {d['excluded']:,}، "
            f"میانگین {value(d['mean'])}، انحراف معیار {value(d['population_sd'])}، "
            f"کران میانه {q['p50']}، کران دامنهٔ میان‌چارکی {d['iqr_bounds']}، "
            f"کران صدک ۹۵ {q['p95']}، صفر {percent(d['zero'])}.")


def markdown(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def category_reports(folder, category, scope, raw, relations, products, cohort, main_policy):
    rev, meta = raw["reviews"], raw["metadata"]
    notes = [f"# {category}", "", f"دامنهٔ اجرا `{scope}`", "",
             "**E01 — پیوند با سؤال اصلی:** آیا متن پنج نقد نخست، علاوه بر امتیاز و زمان همان نقدها، "
             "پیش‌بینی سهم امتیازهای ۱ و ۲ در ده نقد بعدی را بهتر می‌کند؟ "
             "در این فاز فقط توصیف و بررسی روابط انجام می‌شود؛ مقایسهٔ پیش‌بینی S/ST هنوز انجام نشده است. "
             "تعریف دقیق و تأییدشده در [`RESEARCH_CONTRACT_E01_FA.md`](../RESEARCH_CONTRACT_E01_FA.md) آمده است.", ""]
    uni = notes + [
        "## تک‌متغیره", "",
        f"جامعهٔ خام خوانده‌شده {rev['total']:,} شیء نقد و {meta['total']:,} شیء فراداده دارد. "
        "نسخه‌های تکراری در این دو جامعه باقی مانده‌اند. سطرهای خوانده‌شدهٔ خارج از این جمعیت در مانیفست آمده‌اند.", "",
        "### امتیاز و عدم‌توازن", "",
        "سؤال: امتیازها در کدام رده‌ها متمرکزند؟ مخرج فراوانی هر رده، نقدهای با امتیاز معتبر است. "
        "امتیاز ترتیبی است و فاصلهٔ عددی آن فقط برای خلاصه‌سازی استفاده می‌شود.", "",
        f"سهم ۱ یا ۲ ستاره {percent(rev['low_star'])}. {distribution_text(rev['numeric']['rating'])}", "",
    ]
    if rev["numeric"]["rating"]["valid"]:
        mode = max(row["numerator"] for row in rev["numeric"]["rating"]["bins"])
        stars = ", ".join(str(int(row["lower"])) for row in rev["numeric"]["rating"]["bins"]
                          if row["numerator"] == mode)
        uni += [f"رده‌های پرتکرار امتیاز {stars} با {mode:,} نقد برای هر رده.", ""]
    uni += ["### متن، رأی و خرید", "",
            "سؤال: متن تا چه حد موجود است و طول یا شمار رأی چه شکل و دنباله‌ای دارد؟ "
            "طول‌ها فقط برای رشته‌های ناتهی پس از آزمون فاصله محاسبه می‌شوند. صفر رأی در جامعهٔ معتبر می‌ماند.", "",
            "| شاخص | خلاصه و مخرج |", "|---|---|"]
    for key in ("title_chars", "body_chars", "body_words", "helpful_vote"):
        uni.append(f"| `{key}` | {distribution_text(rev['numeric'][key])} |")
    uni += ["", f"عنوان یا بدنهٔ قابل‌استفاده {percent(rev['modalities']['any_text'])} از اشیای نقد. "
            "وضعیت‌های مجزای متن در جدول فراوانی ثبت شده‌اند.", ""]
    for key, row in rev["body_flags"].items():
        uni.append(f"- `{key}` {percent(row)} از بدنه‌های ناتهی. نشانه‌های نویسه‌ای معیار تشخیص زبان نیستند.")
    uni += ["", "- خرید تأییدشده، تأییدنشده و نامعتبر بر کل اشیای نقد:",
            *[f"  - `{r['level']}` {percent(r)}" for r in rev["verified_states"]], "",
            "### زمان، حجم محصول و فراداده", "",
            f"سؤال: نقدهای زمان‌دار و محصولات در کدام دوره‌ها و سطوح حجم متمرکزند؟ "
            f"زمان معتبر {percent(rev['valid_time'])}. محور زمانی سال تقویمی در زمان هماهنگ جهانی است. "
            "شمار سالانه مواجهه یا پوشش ثابت نمونه‌گیری را اندازه نمی‌گیرد.", ""]
    if rev["time_min_utc"] is not None:
        uni += [f"زمان کمینهٔ مشاهده‌شده در UTC برابر `{rev['time_min_utc']}` و بیشینه برابر `{rev['time_max_utc']}` است.", ""]
    if rev["years"]:
        top_year = max(rev["years"], key=lambda row: row["numerator"])
        uni += [f"سال با بیشترین نقد زمان‌دار در محدودهٔ اجرا {top_year['year']}، {percent(top_year)}.", ""]
    uni += [f"حجم خام در واحد محصول در این فایل، با شناسهٔ خام معتبر. {distribution_text(products['raw_review_volume'])}",
            f"حجم پس از اعتبارسنجی هسته، پیش از حذف نسخه‌های تکراری. {distribution_text(products['valid_core_review_volume'])}", ""]
    for key, row in products["raw_volume_groups"].items():
        uni.append(f"- {key} نقد خام برای هر محصول، {percent(row)} از محصولات این فایل.")
    uni += ["", "قیمت، میانگین ستاره و شمار امتیاز از نمای فراداده در زمان گردآوری توصیف می‌شوند. "
            "قیمت صفر باقی می‌ماند و ممکن است نشانهٔ خاص منبع باشد.", "",
            "| شاخص فراداده | خلاصه و مخرج |", "|---|---|"]
    uni += [f"| `{key}` | {distribution_text(d)} |" for key, d in meta["numeric"].items()]
    uni += ["", "نمودار `univariate.png` از فراوانی‌های تجمیعی ساخته شده است. "
            "طبقات توانی طول/رأی/قیمت و حجم محصول فراوانی دقیق در بازه دارند. "
            "کران‌های صدک بازه‌ای‌اند و هیچ نقطهٔ پرت حذف نشده است."]
    markdown(folder / "UNIVARIATE_FA.md", uni)
    multi = notes + [
        "## چندمتغیره", "",
        "### امتیاز، متن، رأی و خرید", "",
        "سؤال: در هر ستاره، چه سهمی متن قابل‌استفاده دارد و طول بدنه، رأی صفر و خرید تأییدشده چه تغییری می‌کنند؟ "
        "مخرج حضور متن و وضعیت خرید همهٔ نقدهای همان ستاره است. مخرج طول و رأی صفر به‌ترتیب بدنه‌های ناتهی و رأی‌های معتبر همان ستاره است. "
        "رأی تجمعی به زمان و فرصت دیده‌شدن وابسته است و برچسب خرید، اصالت نقد را ثابت نمی‌کند.", "",
        f"امتیاز معتبر {percent(relations['valid_stars'])} و "
        f"{relations['excluded_invalid_stars']:,} نقد از این طبقه‌بندی خارج‌اند.", "",
        "| ستاره | عنوان ناتهی | بدنهٔ ناتهی | کران میانهٔ طول بدنه و شمار معتبر/خارج | رأی صفر و شمار خارج | خرید تأییدشده؛ وضعیت نامعتبر |",
        "|---|---|---|---|---|---|"]
    for r in relations["by_star"]:
        d, votes = r["body_length"], r["helpful_vote"]
        multi.append(f"| {r['star']} | {percent(r['title'][-1])} | {percent(r['body'][-1])} | "
                     f"{value(d['quantile_bounds'].get('p50'))}؛ {d['valid']:,}/{d['excluded']:,} | "
                     f"{percent(votes['zero'])}؛ {votes['excluded']:,} | "
                     f"{percent(r['verified'][1])}؛ {r['verified'][2]['numerator']:,} |")
    multi += ["", "### ترکیب ستاره در سال و در سطح حجم محصول", "",
              f"زمان و امتیاز همزمان معتبر برای {percent(relations['valid_rating_and_time'])}. "
              f"خارج از مقایسهٔ زمانی {relations['excluded_invalid_rating_or_time']:,} نقد. "
              "سهم سالانهٔ هر ستاره از کل نقدهای معتبر همان سال است. نمودار نقاط و جدول از دوره‌های مشاهده‌شده استفاده می‌کنند.", ""]
    if relations["by_year"]:
        min_year = min(relations["by_year"], key=lambda r: r["low_star"]["fraction"])
        max_year = max(relations["by_year"], key=lambda r: r["low_star"]["fraction"])
        multi += [f"کمینهٔ مشاهده‌شدهٔ سهم پایین در {min_year['year']} با {percent(min_year['low_star'])} "
                  f"و بیشینه در {max_year['year']} با {percent(max_year['low_star'])}. اندازهٔ مخرج ممکن است بین سال‌ها تفاوت زیادی داشته باشد.", ""]
    multi += ["سؤال حجم: ترکیب امتیاز محصولات کم‌نقد و پرنقد چگونه تفاوت دارد؟ "
              "گروه حجم بر اساس شمار خام و نسبت درون محصول بر اساس نقدهای با امتیاز معتبر همان فایل است. "
              "میانگین با وزن برابر محصول در کنار نسبت نقد-وزن آمده است. وابستگی نقدهای یک محصول و اندازهٔ گروه در تفسیر اهمیت دارد.", "",
              "| بازهٔ حجم | محصول معتبر/خارج | میانگین محصول‌وزن | نسبت نقد‌وزن |", "|---|---|---|---|"]
    for r in products["volume_vs_rating"]:
        multi.append(f"| {r['volume_bin']} | {r['eligible_products']:,}/{r['excluded_products']:,} | "
                     f"{value(r['product_weighted_mean'])} | {percent(r['review_weighted_low_star'])} |")
    join = products["metadata_connection"]
    multi += ["", "### دسترس‌پذیری در اتصال با فراداده", "",
              "سؤال: چند نقد با یک نسخهٔ یکتای فراداده وصل می‌شوند و در این زیرجامعه چه سهمی قیمت یا تصویر محصول دارد؟ "
              "فقط کلید خام مشترک و یک نسخهٔ کانونی مجاز است. تصویر به معنی مرجع نحوی وب است. شمارش اتصال هر نقد حداکثر یک بار است.", "",
              *[f"- `{k}` {percent(r)} از اشیای نقد." for k, r in join["review_states"].items()],
              *[f"- `{k}` {percent(r)} از نقدهای دارای یک نسخهٔ فراداده." for k, r in join["modalities_among_unique_matches"].items()],
              "", "برای مقایسهٔ هم‌زمان چهار دسته و تفاوت دسترس‌پذیری، نمودار مشترک را ببینید. "
              "از اختلاف توصیفی دسته‌ها نمی‌توان سازوکار گمشدگی را تعیین کرد. نمودار دسته در `multivariate.png` است.", ""]
    if cohort:
        main = cohort[main_policy]
        multi += ["## جامعهٔ موقت ۵+۱۰", "",
                  "سؤال: کدام محصولات به ده نقد بعدی می‌رسند و چه تفاوتی با محصولات خارج از پنجره دارند؟ "
                  "هر محصول جهانی یک بار در دستهٔ اصلی وارد می‌شود. افق زمانی نهایی و سیاست اختلاف نسخه‌ها در انتظار شواهد کیفیت‌اند.", "",
                  "| مرحله | عبور از مرحلهٔ قبل | خارج در این مرحله |",
                  "|---|---|---|",
                  *[f"| `{r['stage']}` | {percent(r)} | {r['excluded']:,} |" for r in main["funnel"]],
                  "", "سهم امتیاز پایین در ده نقد آینده در واحد محصول و روی شبکهٔ یازده‌مقداری خلاصه می‌شود. "
                  "زمان سررسید فقط فاصلهٔ نقد پانزدهم تا پنجم است و کاربرد توصیفی دارد.", "",
                  f"- هدف آینده. {distribution_text(main['future_low_share'])}",
                  f"- فاصله بر حسب روز. {distribution_text(main['maturity_days'])}", "",
                  "| علت/وضعیت | محصولات با پنج نقد اولیهٔ نامبهم | سهم امتیاز پایین اولیه | سهم نقدهای متنی اولیه |",
                  "|---|---|---|---|",
                  *[f"| `{k}` | {r['products']:,} | {percent(r['low_star'])} | {percent(r['any_text'])} |"
                    for k, r in main["early_by_outcome"].items()], "",
                  "مخرج دو سهم اولیه پنج برابر تعداد محصولات همان رده با پنجرهٔ اولیهٔ نامبهم است. "
                  "ماتریس شمار امتیاز پایین اولیه و آینده بر تعداد محصولات هر مقدار اولیه نرمال می‌شود و فقط جامعهٔ موقت واجدشرایط را توصیف می‌کند. "
                  "نمودار قیف و سررسید و مقایسهٔ سیاست‌ها در `cohort.png` و اعداد تفکیکی در خروجی جامعهٔ موقت آمده‌اند. "
                  "تفاوت توصیفی هیچ مقایسهٔ پیش‌بینی خارج از نمونه‌ای را انجام نمی‌دهد.", ""]
    else:
        multi += ["## جامعهٔ ۵+۱۰", "", "ترتیب کامل محصول از فایل پیش‌نمایش یا نمونهٔ تأییدنشده معلوم نیست. آمار پنجره در این دامنه منتشر نشده است."]
    time_avail=relations['year_modality_availability']
    multi += ["", "### E37 — وجود متن و سایر وجوه برحسب سال", "",
              "مخرج هر سال تمام نقدهای با زمان معتبر در همان سال است، حتی اگر امتیاز نامعتبر باشد. "
              "ناهمگنی مشاهده‌شده به‌تنهایی سازوکار MCAR/MAR/MNAR را اثبات نمی‌کند.", "",
              "| سال UTC | کل نقد زمان‌دار | متن بدنهٔ ناتهی | عنوان ناتهی | تصویر نقد دارای URL معتبر از نظر نحو | پرچم خرید معتبر |",
              "|---|---:|---|---|---|---|"]
    for r in time_avail['by_year']:
        multi.append(f"| {r['year']} | {r['n']:,} | {percent(r['body_nonblank'])} | "
                     f"{percent(r['title_nonblank'])} | {percent(r['review_image_url'])} | "
                     f"{percent(r['verified_flag_valid'])} |")
    cond=products['metadata_conditioned_ratings']
    multi += ["", "### E38 — رابطهٔ فرادادهٔ یکتای متصل با سهم کم‌ستاره", "",
              "مخرج شرطی: فقط محصولات با دقیقاً یک نسخهٔ کانونی فراداده. هر محصول در نسبت محصول‌وزن "
              "یک بار و در نسبت نقد‌وزن به نسبت تعداد نقدهای دارای امتیاز معتبر نقش دارد. "
              "مشاهدهٔ فراداده در زمان گردآوری است و برای پیش‌بینی در زمان نقد پنجم مجاز تلقی نمی‌شود.", "",
              "| وجه فراداده | وضعیت | کل محصول یکتا | محصول دارای امتیاز معتبر | سهم کم‌ستاره نقد‌وزن | میانگین محصول‌وزن |",
              "|---|---|---:|---|---|---|"]
    for facet,states in cond['facets'].items():
        for label,r in states.items():
            multi.append(f"| {facet} | {label} | {r['all_matched_products']:,} | "
                         f"{percent(r['products_with_valid_ratings'])} | "
                         f"{percent(r['low_star_review_weighted'])} | {value(r['low_star_product_weighted_mean'])} |")
    pair=relations['body_length_vs_helpful_vote']
    multi += ["", "### E39 — رابطهٔ طول متن و رأی مفید", "",
              "ضریب پیرسون تنها برای زوج‌هایی محاسبه شده که هم بدنهٔ ناتهی و هم رأی مفید معتبر دارند. "
              "روی هر دو متغیر تبدیل log(1+x) اعمال می‌شود. نمونهٔ نموداری محدود است؛ ضریب از همهٔ زوج‌های معتبر "
              "در همین دامنه محاسبه می‌شود. وابستگی به محصول/کاربر، عمر نقد و مواجهه کنترل نشده و استنباط علّی مجاز نیست.", "",
              f"همهٔ زوج‌ها: {percent(pair['overall']['valid_pairs'])}، ضریب {value(pair['overall']['correlation'])} "
              f"(وضعیت: {pair['overall']['reason']}).", "",
              "| ستاره | زوج معتبر نسبت به نقدهای همان ستاره | هم‌بستگی | وضعیت |",
              "|---|---|---|---|"]
    for r in pair['by_star']:
        multi.append(f"| {r['star']} | {percent(r['valid_pairs'])} | "
                     f"{value(r['correlation'])} | {r['reason']} |")
    multi += ["", "نمودارهای تکمیلی در `multivariate_extended.png` و مقادیر کامل در `multivariate.json` آمده‌اند.", ""]
    markdown(folder / "MULTIVARIATE_FA.md", multi)
