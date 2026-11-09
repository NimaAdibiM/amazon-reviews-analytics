# دریافت داده‌ها

داده‌های خام `Amazon Reviews’23` از منبع منتشرشده توسط McAuley Lab, UC San Diego دریافت می‌شوند: https://amazon-reviews-2023.github.io/

در اجرای محلی ساختار ورودی زیر مورد انتظار است:

```text
data/raw/Electronics.jsonl.gz
data/raw/meta_Electronics.jsonl.gz
data/raw/Beauty_and_Personal_Care.jsonl.gz
data/raw/meta_Beauty_and_Personal_Care.jsonl.gz
data/raw/Appliances.jsonl.gz
data/raw/meta_Appliances.jsonl.gz
data/raw/All_Beauty.jsonl.gz
data/raw/meta_All_Beauty.jsonl.gz
```

نشانی دقیق دریافت هشت فایل نیز در `reports/phase1/quality/RUN_MANIFEST.json` و خلاصه‌های کیفیت ثبت شده است. برای رعایت حجم مخزن و بازنشر مسئولانه، فایل‌های خام و پایگاه‌های میانی در مخزن عمومی قرار نمی‌گیرند. ثبت تاریخ و روش واقعی دریافت باید از سوابق محلی انجام شود؛ تاریخ دانلود از روی تاریخ فایل حدس زده نمی‌شود.
