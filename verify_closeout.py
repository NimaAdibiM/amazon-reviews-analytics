"""Diff new EDA output against retained Batch 6 synthetic baseline.

This verifies preservation of existing evidence, not full-scale real-data validity.
Usage: python -m audit_lab.verify_closeout --new RUN --out REPORT.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINALS=('univariate.json','multivariate.json','cohort.json')
CATS=('All_Beauty','Appliances','Beauty_and_Personal_Care','Electronics')


def compare_prior(old, new, path, problems):
    count=0
    if isinstance(old,dict):
        if not isinstance(new,dict):
            problems.append({'path':path,'kind':'object_type_changed'});return count
        for key,value in old.items():
            if key not in new:
                problems.append({'path':path+'.'+key,'kind':'previous_field_removed'})
            else:
                count+=compare_prior(value,new[key],path+'.'+key,problems)
    elif isinstance(old,list):
        if not isinstance(new,list) or len(new)!=len(old):
            problems.append({'path':path,'kind':'prior_array_length_or_type_changed'});return count
        for i,(a,b) in enumerate(zip(old,new)):
            count+=compare_prior(a,b,path+f'[{i}]',problems)
    else:
        count+=1
        if old!=new or type(old) is not type(new):
            problems.append({'path':path,'kind':'previous_value_changed','old':old,'new':new})
    return count


def inspect(old_root,new_root):
    problems=[]; checked=0; images=0; matched_images=0; baseline_json=0
    for cat in CATS:
        for filename in ORIGINALS:
            a=old_root/cat/filename;b=new_root/cat/filename
            if not a.is_file() or not b.is_file():
                problems.append({'path':cat+'/'+filename,'kind':'missing_comparable_json'});continue
            checked+=compare_prior(json.loads(a.read_text(encoding='utf-8')),
                                   json.loads(b.read_text(encoding='utf-8')),
                                   f'{cat}/{filename}',problems)
            baseline_json+=1
        for p in (old_root/cat).glob('*.png'):
            images+=1;other=new_root/cat/p.name
            if other.is_file() and hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(other.read_bytes()).digest():
                matched_images+=1
            else:
                problems.append({'path':cat+'/'+p.name,'kind':'old_chart_changed'})
    for name in ('global_products.json','category_by_star.json','category_comparison.json','cohort_policy_comparison.json'):
        a=old_root/name;b=new_root/name
        if a.is_file() and b.is_file():
            checked+=compare_prior(json.loads(a.read_text(encoding='utf-8')),
                                   json.loads(b.read_text(encoding='utf-8')),name,problems)
            baseline_json+=1
    return {'status':'PASS_SYNTHETIC_REGRESSION' if not problems else 'FAIL_SYNTHETIC_REGRESSION',
            'previous_scalar_values_checked':checked,
            'prior_images_checked':images,'prior_images_identical':matched_images,
            'prior_json_files_checked':baseline_json,'failures':problems,
            'warning':'Old fields and figures checked on one identical synthetic fixture, not raw real Amazon data.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--new',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--baseline',type=Path,default=Path(__file__).resolve().parent/'evidence'/'batch6'/'synthetic_output')
    args=p.parse_args()
    result=inspect(args.baseline,args.new)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(result['status'],result['previous_scalar_values_checked'],result['prior_images_identical'],
          'of',result['prior_images_checked'])
    return 0 if result['status'].startswith('PASS') else 1

if __name__=='__main__':raise SystemExit(main())
