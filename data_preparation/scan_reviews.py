                                                   
import argparse
from pathlib import Path

from ingest import ingest_category
from metadata import CATEGORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for category in CATEGORIES:
        audit = ingest_category(args.data / f"{category}.jsonl", category, args.out / category,
                                workers=args.workers, force=args.force)
        print(category, audit["counts"], flush=True)


if __name__ == "__main__":
    main()
