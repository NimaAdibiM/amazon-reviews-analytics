# داده‌های فاز دوم

در جدول دسته‌های [Amazon Reviews’23](https://amazon-reviews-2023.github.io/#grouped-by-category)، برای هر یک از دسته‌های `All_Beauty`، `Appliances`، `Beauty_and_Personal_Care` و `Electronics` دو فایل review و meta دریافت کنید. فایل‌ها را پس از استخراج در `data/raw/` قرار دهید. نام‌های مورد انتظار `<Category>.jsonl` و `meta_<Category>.jsonl` هستند. فایل `reports/phase2_evidence/input_hashes.json` هش فایل‌های استخراج‌شدهٔ اجرای مرجع را ثبت می‌کند.

خط لوله از محصول والد در هر دسته یک نمونه می‌سازد. ورودی پنج نقد معتبر متمایز اول و برچسب سهم نقدهای ۱ یا ۲ ستاره در ده نقد بعدی است. تقسیم برای هر والد با SHA‑256 و بذر ۴۲ انجام می‌شود. نمونهٔ خوشه‌بندی شامل ۱۲۸ محصول آموزشی از هر دسته است.

منبع انتشار، McAuley Lab در UC San Diego است. ارجاع پژوهشی [Hou et al., Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952).
