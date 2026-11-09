"""Guided, non-destructive GitHub phase-1 publication, requiring explicit approval."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
PATHS=['README.md','PROJECT_CARD.md','AI_USAGE.md','requirements.txt','.gitignore',
 'reports','docs','data/README.md','structure_audit.py','quality_audit.py',
 'phase1_eda_delivery','tools','RUN_EDA_PHASE1_ALL_CATEGORIES.cmd',
 'CHECK_EDA_PHASE1_INPUTS.cmd','TEST_EDA_PHASE1_DELIVERY.cmd','PUBLISH_PHASE1_TO_GITHUB.cmd',
 'IMPORT_EDA_RESULTS.cmd','START_HERE_BEFORE_GITHUB_FA.md']
def run(*args,check=True):
    proc=subprocess.run(['git',*args],cwd=ROOT,text=True,check=False)
    if check and proc.returncode:raise SystemExit('Git failed: '+' '.join(args))
    return proc.returncode

def yes(question):
    return input(question+' Type YES to continue: ').strip()=='YES'

def main():
    if not (ROOT/'.git').exists():raise SystemExit('Not a Git repository root')
    print('Remote:');run('remote','-v')
    print('Current branch:');run('branch','--show-current')
    print('Status BEFORE staging:');run('status','--short')
    if not yes('Stage the standard Phase 1 files (no raw data)?'):return
    run('add','--',*PATHS)
    print('Staged change summary:');run('diff','--cached','--stat')
    if run('diff','--cached','--quiet',check=False)==0:
        print('No staged changes. Will not create a new commit.')
        return
    if not yes('Create a commit containing these staged changes?'):
        print('Files remain staged. No commit or push performed.')
        return
    run('commit','-m','Phase 1: verified data audits and exploratory analysis')
    if not yes('Push this branch to origin?'):
        print('Commit exists locally; not pushed.')
        return
    run('push','origin','HEAD')
    if run('tag','-l','phase-1')==0:
        tags=subprocess.run(['git','tag','-l','phase-1'],cwd=ROOT,capture_output=True,text=True)
        if tags.stdout.strip():
            print('Tag phase-1 already exists. No modifications made. Inspect it manually.')
            return
    if yes('Create phase-1 tag at the just-pushed commit and push it?'):
        run('tag','phase-1');run('push','origin','phase-1')
        print('Branch and phase-1 tag were submitted to origin.')
if __name__=='__main__':main()
