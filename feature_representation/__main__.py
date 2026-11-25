import argparse
import json
from pathlib import Path

from .common import atomic_json, environment, read_json


def main():
    parser=argparse.ArgumentParser(prog='python -m feature_representation')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run'); run.add_argument('--config',required=True); run.add_argument('--output',required=True)
    run.add_argument('--authorize-full-run',action='store_true')
    consume=sub.add_parser('consume'); consume.add_argument('--input',required=True)
    export=sub.add_parser('export-reviews'); export.add_argument('--cohorts',nargs='+',required=True); export.add_argument('--output',required=True); export.add_argument('--max-products',type=int,default=4000)
    sample=sub.add_parser('sample'); sample.add_argument('--source',required=True); sample.add_argument('--cohorts',nargs='+',required=True); sample.add_argument('--output',required=True)
    sample.add_argument('--per-category',type=int,default=200); sample.add_argument('--max-scan-records',type=int,default=1000000)
    sample.add_argument('--seed',type=int,default=42)
    model=sub.add_parser('prepare-model'); model.add_argument('--model',choices=['bge','e5'],required=True); model.add_argument('--output',required=True); model.add_argument('--max-bytes',type=int,default=1500000000)
    plots=sub.add_parser('plot'); plots.add_argument('--input',required=True); plots.add_argument('--output',required=True)
    diag=sub.add_parser('diagnose'); diag.add_argument('--config',required=True); diag.add_argument('--output',required=True)
    diag.add_argument('--authorize-full-run',action='store_true')
    pairs=sub.add_parser('evaluate-pairs'); pairs.add_argument('--input',required=True); pairs.add_argument('--pairs',required=True); pairs.add_argument('--output',required=True)
    sub.add_parser('environment')
    args=parser.parse_args()
    if args.command=='run':
        from .pipeline import run_pipeline
        m=run_pipeline(read_json(args.config),args.output,args.authorize_full_run)
        result={'status':m['status'],'n':m['n_products'],'level':m['metadata']['level'],'output':args.output}
    elif args.command=='consume':
        from .artifacts import load_stage_a_artifacts
        n=0; spaces=set()
        for keys,arrays in load_stage_a_artifacts(args.input):
            n+=len(keys); spaces.update(arrays)
        result={'verified_rows':n,'spaces':sorted(spaces),'recomputed_embeddings':False}
    elif args.command=='export-reviews':
        from .contracts import export_reviews
        result={'n':export_reviews(args.cohorts,args.output,args.max_products)}
    elif args.command=='sample':
        from .sampling import extract_train_samples
        result=extract_train_samples(args.source,args.cohorts,args.output,args.per_category,seed=args.seed,max_scan_records=args.max_scan_records)
    elif args.command=='prepare-model':
        from .semantic import prepare_model
        result=prepare_model(args.model,args.output,args.max_bytes)
    elif args.command=='plot':
        from .plots import plot_run
        result=plot_run(args.input,args.output)
    elif args.command=='diagnose':
        from .experiments import run_diagnostics
        result=run_diagnostics(read_json(args.config),args.output,args.authorize_full_run)
    elif args.command=='evaluate-pairs':
        from .experiments import evaluate_pairs
        result=evaluate_pairs(args.input,args.pairs,args.output)
    else:
        result=environment()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
