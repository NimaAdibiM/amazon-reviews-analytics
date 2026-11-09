"""Deterministic invented records; generate exactly eight local input files."""
import gzip
import json
from pathlib import Path
import sys


CATEGORIES = ("All_Beauty", "Appliances", "Beauty_and_Personal_Care", "Electronics")
DAY = 86_400_000
ORIGIN = 1_577_836_800_000


def review(parent, i, when=None):
    return {
        "parent_asin": parent, "asin": f"{parent}-variant", "user_id": f"u{i}",
        "timestamp": ORIGIN + DAY*(i if when is None else when),
        "rating": (i-1) % 5 + 1, "title": "" if i % 2 else "review headline",
        "text": ("  " if i % 4 == 0 else "ساختگی �" if i % 3 == 0 else "a synthetic review"),
        "helpful_vote": 0 if i % 3 else 2**i, "verified_purchase": i % 2 == 1,
        "images": [{"small_image_url": "https://example.invalid/i"}] if i % 3 == 1 else [],
    }


def product(parent, n, ties=(), offset=0):
    return [review(parent, i, offset+i-1 if i in ties else offset+i) for i in range(1, n+1)]


def meta(parent, price=10):
    return {"parent_asin": parent, "price": price, "average_rating": 4.3, "rating_number": 10,
            "title": "Invented", "images": [{"thumb": "https://example.invalid/product"}]}


def generate(folder, empty=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    shared = product("shared", 16)
    all_reviews = shared[:10] + [dict(reversed(list(shared[0].items())))] + product("short", 5)
    all_reviews += [
        dict(review("bad_rating", 1), rating=True),
        dict(review("bad_rating", 2), rating=5.5),
        dict(review("bad_rating", 3), rating=0),
        dict(review("bad_rating", 4), rating=6),
        dict(review("bad_rating", 5), rating="1"),
        dict(review("bad_time", 1), timestamp=float(ORIGIN)),
        dict(review("bad_time", 2), timestamp=10**50),
        dict(review("bad_time", 3), timestamp=True),
        dict(review("A", 1), parent_asin="   "),
        dict(review("A", 1), parent_asin=42),
        dict(review("A", 1), parent_asin="\ud800"),
        dict(review(" spaced ", 1), rating=1.0, verified_purchase=1, helpful_vote=1.0, text=17),
    ]
    b = product("conflict", 17, offset=365)
    b.append(dict(b[-1], text="conflicting version after the first 16", rating=1))
    # Stable same-time order is only for ties wholly inside one window.
    e = product("inside", 16, ties=(2,))
    n = [dict(row, title="", text=None) for row in product("no_text", 15, offset=720)]
    data = {
        "All_Beauty": (all_reviews, [meta("shared"), meta("shared"), meta("short", 0),
                                    meta("spaced", 5), meta("orphan", "10")]),
        "Appliances": (shared[8:] + product("at_five", 15, (6,)) + product("exactly15", 15),
                       [meta("shared"), meta("shared", 11), meta("shared", 11), meta("at_five")]),
        "Beauty_and_Personal_Care": (b + product("at_fifteen", 16, (16,)) + product("both", 16, (6, 16))
                                      + product("few_with_tie", 6, (6,)),
                                     [meta("conflict"), meta("at_fifteen"), meta("both")]),
        "Electronics": (e + n, [meta("inside"), meta("no_text")]),
    }
    config = dict(project_root=".", input_role="synthetic", timestamp_unit="ms", batch_size=7,
                  progress_every_lines=0, datasets={})
    for i, (cat, pair) in enumerate(data.items()):
        files = {}
        for j, (role, rows) in enumerate(zip(("reviews", "metadata"), pair)):
            name = ("" if j == 0 else "meta_") + cat + ".jsonl" + (".gz" if (i+j) % 2 else "")
            path = folder / name
            lines = [json.dumps(row, ensure_ascii=True).encode("utf-8") + b"\n" for row in rows] if not empty else []
            if not empty and cat == "All_Beauty" and role == "reviews":
                lines += [b"{oops}\n", b"[]\n", b"\n", b'{"rating":NaN}\n',
                          b'{"rating":1e999}\n', b'{"rating":1,"rating":5}\n', b"\xff\n"]
            if path.suffix == ".gz":
                path.write_bytes(gzip.compress(b"".join(lines), mtime=0))
            else:
                path.write_bytes(b"".join(lines))
            files[f"{role}_path"] = name
        config["datasets"][cat] = files
    cfg_path = folder / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return cfg_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m tests.fixture NEW_DIRECTORY")
    print(generate(Path(sys.argv[1])))
