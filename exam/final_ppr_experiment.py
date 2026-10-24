"""Supervise a bounded final experiment; publish only fully verified checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .evaluation import average_precision_at_5
else:
    from evaluation import average_precision_at_5


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read_predictions(path, users, candidates):
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        if next(reader, None) != ['user_id','tv_show_id']:
            raise ValueError('Expected exact submission header')
        rows = list(reader)
    if len(users) != 1260 or len(set(users)) != 1260 or any(len(r) != 2 for r in rows) or [r[0] for r in rows] != list(users):
        raise ValueError('Expected exact 1,260-user template order')
    pool = set(candidates)
    result = {}
    for user, value in rows:
        if not user.isascii() or not user.isdecimal() or int(user) <= 0 or str(int(user)) != user:
            raise ValueError('Invalid exact user ID')
        items = tuple(value.split(' '))
        average_precision_at_5((), items)
        if any(str(int(i)) != i for i in items) or not set(items) <= pool:
            raise ValueError('Noncanonical or ineligible prediction')
        result[user] = items
    return result


def verify_checkpoint(folder, context, result=None):
    if result is None:
        result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if result.get('verified') is not True or not math.isfinite(result.get('map_at_5', float('nan'))):
        raise ValueError('Checkpoint is not verified')
    required = {'submission.csv','validation-predictions.csv','tie-summary.json'}
    if not required <= result['artifacts'].keys():
        raise ValueError('Checkpoint artifacts incomplete')
    for name, expected in result['artifacts'].items():
        if Path(name).name != name or file_hash(folder/name) != expected:
            raise ValueError(f'Checkpoint artifact hash mismatch: {name}')
    if file_hash(folder/'submission.csv') != result['csv_sha256']:
        raise ValueError('Submission hash mismatch')
    read_predictions(folder/'submission.csv', context['users'], context['future_candidates'])
    predictions = read_predictions(folder/'validation-predictions.csv', context['users'], context['validation_candidates'])
    scores = {u:average_precision_at_5(context['targets'][u], predictions[u]) for u in context['users']}
    primary = math.fsum(scores.values())/len(scores)
    auxiliary = context['auxiliary_users']
    auxiliary_map = math.fsum(scores[u] for u in auxiliary)/len(auxiliary) if auxiliary else None
    if not math.isclose(primary, result['map_at_5'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Checkpoint MAP does not reproduce')
    if ((auxiliary_map is None) != (result['auxiliary_map_at_5'] is None)
            or auxiliary_map is not None and not math.isclose(auxiliary_map, result['auxiliary_map_at_5'], rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Checkpoint auxiliary MAP does not reproduce')
    if result['cohorts'] != {'all_users':len(scores),'at_least_five_coverage80':len(auxiliary)}:
        raise ValueError('Checkpoint cohort counts differ')
    ties = json.loads((folder/'tie-summary.json').read_text(encoding='utf-8'))
    if ties['completed_seeds'] != list(range(20)):
        raise ValueError('All 20 saved tie resolutions must finish before promotion')
    return result


def publish_selection(output, completed, context):
    """Worker-only atomic commit; the CSV lives in an immutable completed directory."""
    best = max(completed, key=lambda r:(r['map_at_5'], -r.get('sequence',0)))
    folder = Path(output)/'completed'/best['configuration']
    # This potentially expensive verification runs inside the killable worker.
    verify_checkpoint(folder, context, best)
    selection = {**best, 'status':'selected_verified_configuration',
                 'completed_configurations':[r['configuration'] for r in completed],
                 'completed_results':completed,
                 'selection_rule':'Highest full-cohort frozen-target MAP; exact ties keep earlier completed configuration. Sensitivity never selects the winner.',
                 'submission_path':str((folder/'submission.csv').resolve())}
    fields = ['configuration','damping','restart','map_at_5','auxiliary_map_at_5','csv_sha256','sequence']
    temporary = Path(output)/'results.csv.tmp'
    with temporary.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(completed)
    os.replace(temporary,Path(output)/'results.csv')
    # Sole commit marker. A timeout before this leaves the previous selection valid.
    atomic_json(Path(output)/'selection.json',selection)
    return selection


def supervise(command, output, budget_seconds, *, started=None):
    """Supervise only process/time; heavy validation/export stays in the bounded worker."""
    output = Path(output)
    started = time.monotonic() if started is None else started
    reserve = min(5., budget_seconds/4)
    work_deadline = started+budget_seconds-reserve
    expired = False
    with (output/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while process.poll() is None:
                remaining=work_deadline-time.monotonic()
                if remaining<=0:
                    expired=True
                    process.terminate()
                    try:process.wait(timeout=min(1.,reserve/2))
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=min(1.,reserve/2))
                    break
                try:process.wait(timeout=min(.1,remaining))
                except subprocess.TimeoutExpired:pass
        finally:
            if process.poll() is None:process.kill()
    status_path=output/'worker-status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    expired=expired or status.get('state')=='budget_expired'
    selection_path=output/'selection.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else None
    run_status={'budget_seconds':budget_seconds,'worker_stop_reserve_seconds':reserve,'budget_expired':expired,
                'worker_exit_code':process.returncode,'worker_status':status,
                'elapsed_seconds':round(time.monotonic()-started,3),
                'configuration':selection['configuration'] if selection else None,
                'csv_sha256':selection['csv_sha256'] if selection else None,
                'status':'verified_selection_available' if selection else 'no_verified_configuration'}
    atomic_json(output/'run-status.json',run_status)
    if selection is None:
        raise RuntimeError('No verified configuration completed; original submissions remain unchanged. Inspect worker.log.')
    print(f"Selected {selection['configuration']}; CSV: {selection['submission_path']}; SHA-256: {selection['csv_sha256']}",flush=True)
    return {**selection,**run_status}


def main():
    started = time.monotonic()
    exam = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Eight-minute final PPR experiment with verified checkpoint fallback and separate Kaggle export. No submission upload.')
    parser.add_argument('--exam-dir',type=Path,default=exam)
    parser.add_argument('--output-dir',type=Path,default=exam/'derived'/'ppr-final-8min')
    parser.add_argument('--budget-seconds',type=float,default=480.)
    args=parser.parse_args()
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 480:
        parser.error('budget-seconds must be positive and at most 480')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    print(f'Budget: {args.budget_seconds:g} seconds including preparation and verification. Progress log: {args.output_dir/"worker.log"}',flush=True)
    deadline=started+args.budget_seconds-min(5.,args.budget_seconds/4)
    command=[sys.executable,'-B',str(exam/'final_ppr_worker.py'),'--exam-dir',str(args.exam_dir.resolve()),
             '--output-dir',str(args.output_dir.resolve()),'--deadline',str(deadline)]
    supervise(command,args.output_dir,args.budget_seconds,started=started)


if __name__=='__main__':main()
