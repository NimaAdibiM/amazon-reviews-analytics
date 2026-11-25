from collections import Counter
import json
import sqlite3

import numpy as np

from .common import key_tuple,object_hash,require_train


def create_index(connection):
    connection.execute('CREATE TABLE members(category TEXT,parent TEXT,split TEXT,rank TEXT,PRIMARY KEY(category,parent))')
    connection.execute('CREATE INDEX category_rank ON members(category,rank,parent)')
    connection.execute('CREATE INDEX parent_index ON members(parent)')


def add_key(connection,row,seed):
    connection.execute('INSERT INTO members VALUES(?,?,?,?)',(row['category'],row['parent_asin'],row['split'],object_hash([seed,row['parent_asin']])))


def choose_groups(connection,strategy,max_rows,categories):
    if max_rows<1 or strategy not in ('natural','balanced') or not categories:
        raise ValueError('INVALID_GROUP_SAMPLE')
    selected=set(); counts=Counter();selected_rows=0
    selected_categories=tuple(sorted(set(categories)))
    placeholders=','.join('?' for _ in selected_categories)
    def members(parent):
        return connection.execute(f'SELECT category FROM members WHERE parent=? AND category IN ({placeholders}) ORDER BY category',(parent,*selected_categories)).fetchall()
    if strategy=='natural':
        stream=connection.execute(f'SELECT parent,MIN(rank) FROM members WHERE category IN ({placeholders}) GROUP BY parent ORDER BY MIN(rank),parent',selected_categories)
        for parent,_ in stream:
            group=members(parent)
            if selected_rows+len(group)>max_rows:
                break
            selected.add(parent);selected_rows+=len(group)
            counts.update(c for c, in group)
    else:
        streams={c:iter(connection.execute('SELECT parent FROM members WHERE category=? ORDER BY rank,parent',(c,))) for c in selected_categories}
        active=set(selected_categories)
        while active and selected_rows<max_rows:
            category=min(active,key=lambda c:(counts[c],c))
            candidate=next(streams[category],None)
            if candidate is None:
                active.remove(category);continue
            parent=candidate[0]
            if parent in selected:
                continue
            group=members(parent)
            if selected_rows+len(group)>max_rows:
                continue
            selected.add(parent);selected_rows+=len(group)
            counts.update(c for c, in group)
    rows=[]
    for parent in selected:
        rows.extend({'category':c,'parent_asin':parent,'split':s} for c,s in connection.execute(f'SELECT category,split FROM members WHERE parent=? AND category IN ({placeholders})',(parent,*selected_categories)))
    rows.sort(key=key_tuple)
    return rows,{'strategy':strategy,'capacity':max_rows,'unused_capacity':max_rows-len(rows),'capacity_note':'Natural preserves the hash-ordered complete-group prefix and may leave usable capacity unfilled; balanced skips oversized candidates without splitting parents','category_counts':dict(counts),'n_unique_parents':len(selected),
        'group_closure':'All available selected-category rows of a selected global parent are retained',
        'inclusion_probability':'NOT_CLOSED_FORM_UNDER_ROW_CAP_AND_BALANCE; no inverse-probability or population-CI claim',
        'balance':'greedy category deficit with complete parent groups; exact equal counts not guaranteed' if strategy=='balanced' else 'uniform hashed parent priority, complete-group prefix under row cap; not a fixed-size row SRS'}


def sample_keys(keys,strategy,max_rows,seed,categories=None):
    require_train(keys)
    with sqlite3.connect(':memory:') as connection:
        create_index(connection)
        for row in keys:add_key(connection,row,seed)
        return choose_groups(connection,strategy,max_rows,categories or sorted({r['category'] for r in keys}))


def subsample_indices(keys,fraction,seed):
    require_train(keys)
    parents=sorted({r['parent_asin'] for r in keys},key=lambda p:object_hash([seed,p]))
    n=max(1,int(len(parents)*fraction))
    selected=set(parents[:n])
    return np.asarray([i for i,r in enumerate(keys) if r['parent_asin'] in selected],dtype=int)


def approved_subsample_indices(keys,fraction,seed):
    require_train(keys)
    if not 0<fraction<1:raise ValueError('INVALID_SUBSAMPLE_FRACTION')
    parents=np.array(sorted({r['parent_asin'] for r in keys}))
    n=int(np.ceil(len(parents)*fraction))
    selected=set(np.random.default_rng(seed).choice(parents,size=n,replace=False))
    return np.asarray([i for i,r in enumerate(keys) if r['parent_asin'] in selected],dtype=int)
