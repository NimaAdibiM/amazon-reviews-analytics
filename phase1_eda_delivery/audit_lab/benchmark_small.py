"""Small, reproducible SQLite indexing pilot; not a 70M-scale guarantee."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc
import platform
from eda.index import Index


def measure(n=10000, batch=250):
    with TemporaryDirectory() as folder:
        path=Path(folder)/'index.sqlite'
        index=Index(path,True)
        tracemalloc.start()
        start=perf_counter()
        for i in range(n):
            row={'parent_asin':f'p{i:08}', 'user_id':f'u{i:08}',
                 'timestamp':1577836800000+i,'rating':i%5+1,
                 'text':'deterministic synthetic content'}
            index.review('Electronics',row,row['parent_asin'],row['user_id'],row['timestamp'],row['rating'],True,2020)
            if (i+1)%batch==0:index.flush()
        index.flush()
        seconds=perf_counter()-start
        live,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows=index.db.execute('SELECT COUNT(*) FROM review').fetchone()[0]
        parents=index.db.execute('SELECT COUNT(*) FROM membership').fetchone()[0]
        bytes_on_disk=path.stat().st_size
        index.db.close()
    return {'scope':'small_synthetic_SQLite_index_pilot','records_requested':n,
            'distinct_review_rows':rows,'unique_parent_keys':parents,
            'index_size_bytes_at_boundary':bytes_on_disk,
            'index_size_bytes_per_review_observed':bytes_on_disk/n if n else None,
            'tracemalloc_peak_traced_bytes':peak,'tracemalloc_live_traced_bytes':live,
            'elapsed_seconds':seconds,'python':platform.python_version(),
            'limitations':['tracemalloc is not process peak RSS','data is much smaller and simpler than Amazon source',
                           'SQLite indexing costs are data-dependent and cannot be linearly projected to full scale',
                           'no Python list stores all reviews but the disk index grows with unique records']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=measure();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
