"""Worker for final_ppr_experiment; immutable verified checkpoints, training only."""
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

if __package__:
    from .als import build_interactions
    from .diagnose_labels import _rows, _save, _close
    from .evaluate_ppr_counts import _unique_training_rows, load_graph, load_targets, TRAIN_WINDOW, VALIDATION_WINDOW
    from .evaluation import candidate_programmes, PopularityModel
    from .generate_kaggle import _template_users, _training_metadata, FUTURE_START, FUTURE_END, validate_submission
    from .ppr import PPRConfig, personalized_pagerank
    from .tie_sensitivity import resolve_targets, score_methods, SEEDS
    from .final_ppr_experiment import atomic_json, file_hash, read_predictions, verify_checkpoint, publish_selection
else:
    from als import build_interactions
    from diagnose_labels import _rows, _save, _close
    from evaluate_ppr_counts import _unique_training_rows, load_graph, load_targets, TRAIN_WINDOW, VALIDATION_WINDOW
    from evaluation import candidate_programmes, PopularityModel
    from generate_kaggle import _template_users, _training_metadata, FUTURE_START, FUTURE_END, validate_submission
    from ppr import PPRConfig, personalized_pagerank
    from tie_sensitivity import resolve_targets, score_methods, SEEDS
    from final_ppr_experiment import atomic_json, file_hash, read_predictions, verify_checkpoint, publish_selection


class BudgetExpired(TimeoutError):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired('Worker deadline reached; retain completed verified checkpoints')


class FastRanker:
    """Same supported-item-first ranking, cached popularity/ID ties and stable sorting."""
    def __init__(self, item_ids, popularity, candidates):
        self.item_ids = tuple(item_ids)
        self.pool = frozenset(candidates)
        self.order = popularity.rank(self.pool)
        self.fallback = popularity.top_five(self.pool)
        index = {item:i for i,item in enumerate(self.item_ids)}
        self.supported = tuple(item for item in self.order if item in index)
        self.indices = np.asarray([index[item] for item in self.supported],dtype=np.int64)

    def rank(self, scores):
        scores = np.asarray(scores,dtype=np.float64)
        if scores.shape != (len(self.item_ids),) or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError('Invalid score vector')
        positions = np.argsort(-scores[self.indices],kind='stable')[:5]
        result = [self.supported[i] for i in positions]
        for item in self.order:
            if len(result) == 5:break
            if item not in result:result.append(item)
        return tuple(result)


@dataclass(frozen=True)
class RestartProfile:
    user_ids: tuple
    item_ids: tuple
    by_user: tuple
    popularity: object


def strict_profile(data, high_counts):
    index = {item:i for i,item in enumerate(data.item_ids)}
    grouped = {}
    for (user,item),count in high_counts.items():
        if user not in data.popularity.training_users or item not in index or count <= 0:
            raise ValueError('Strict counts must come from the original training graph')
        grouped.setdefault(user,[]).append((index[item],count))
    users = tuple(u for u in data.user_ids if u in grouped)
    by_user = tuple((np.array([i for i,_ in sorted(grouped[u])],dtype=np.int64),
                     np.array([n for _,n in sorted(grouped[u])],dtype=np.float64)) for u in users)
    return RestartProfile(users,data.item_ids,by_user,data.popularity)


def run_stages(baseline, trial):
    completed = [baseline]
    for damping in (.05,.10,.25):
        result = trial(damping,'all_counts')
        if result is not None:completed.append(result)
    winner = max(completed,key=lambda r:r['map_at_5'])
    result = trial(winner['damping'],'coverage80_counts')
    if result is not None:completed.append(result)
    return completed


def configuration_name(damping, restart):
    return f"{'counts' if restart == 'all_counts' else 'coverage80'}-d{round(damping*100):03d}"


def prepare(exam, output, deadline):
    """Verify frozen artifacts and read original training counts once; never fit validation."""
    derived=exam/'derived'; run=derived/'ppr-count-restart-050'; diagnostic=derived/'label-diagnostic'
    prior=derived/'kaggle-ppr-count-restart-050'; graph_dir=derived/'ppr-damping-050'
    source=exam/'dataset-sweet-tv'; matching=derived/'matching'
    source_audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    submission_audit=json.loads((prior/'generation-audit.json').read_text(encoding='utf-8'))
    if (source_audit.get('restart_weighting') != 'watch_counts' or source_audit.get('parameters') != asdict(PPRConfig(damping=.5))
            or source_audit.get('training_window') != TRAIN_WINDOW or source_audit.get('validation_window') != VALIDATION_WINDOW
            or not source_audit.get('saved_graph_reused') or not source_audit.get('input_hashes_unchanged')):
        raise ValueError('Expected completed original count-restart .50 run')
    # Include actual consumed files, plus exact old provenance inputs, in new evidence.
    expected={Path(r['path']).resolve():r['sha256'] for r in source_audit['inputs']}
    for r in submission_audit['inputs']:
        p=Path(r['path']).resolve()
        if p in expected and expected[p] != r['sha256']:raise ValueError('Saved run/submission provenance differs')
        expected[p]=r['sha256']
    additional=[run/n for n in ('audit.json','evaluation.json','ppr-scores.npz','ppr-diagnostics.csv','validation-predictions.csv','tie-targets.csv','tie-sensitivity.csv')]
    additional += [prior/n for n in ('generation-audit.json','submission.csv','candidate-popularity.csv')]
    additional += [diagnostic/'audit.json',diagnostic/'labels.csv',diagnostic/'qualifying-frequencies.csv',source/'submission.csv',source/'export_arh_31-42-final.csv']
    hashes={}
    for p in dict.fromkeys(list(expected)+additional):
        check_deadline(deadline);h=file_hash(p)
        if p in expected and expected[p] != h:raise ValueError(f'Changed frozen input: {p}')
        hashes[p]=h
    if file_hash(prior/'submission.csv') != submission_audit['csv_sha256']:raise ValueError('Existing submission hash mismatch')
    users=_template_users(source/'submission.csv')
    future_support={r['tv_show_id']:int(r['training_distinct_users']) for r in _rows(prior/'candidate-popularity.csv',['tv_show_id','training_distinct_users'],True)}
    future=frozenset(future_support)
    validation_support={r['tv_show_id']:int(r['training_distinct_users']) for r in _rows(graph_dir/'candidate-popularity.csv',['tv_show_id','training_distinct_users'],True)}
    graph_rows=list(_rows(graph_dir/'graph-items.csv',['node_index','tv_show_id','training_distinct_users']))
    graph_items=tuple(r['tv_show_id'] for r in graph_rows)
    if len(set(graph_items))!=len(graph_items) or [int(r['node_index']) for r in graph_rows]!=list(range(len(graph_items))):
        raise ValueError('Saved graph node order invalid')
    counts={r['tv_show_id']:int(r['training_distinct_users']) for r in graph_rows}
    if any(n<=0 for n in counts.values()):raise ValueError('Saved graph popularity support invalid')
    for support in (future_support,validation_support):
        if any(counts.get(i,0)!=n for i,n in support.items()):raise ValueError('Cached fallback support differs')
    # The already verified CSV and unchanged source hashes permit a ready baseline
    # before scanning training/future rows again for additional trials.
    if (submission_audit.get('restart_weighting')!='watch_counts' or submission_audit.get('parameters')!=asdict(PPRConfig(damping=.5))
            or submission_audit.get('training_window')!=TRAIN_WINDOW
            or not submission_audit.get('saved_validation_rankings_reproduced')
            or submission_audit.get('csv_validation',{}).get('users')!=1260):
        raise ValueError('Expected previously verified count-restart future submission')
    diagnostic_audit=json.loads((diagnostic/'audit.json').read_text(encoding='utf-8'))
    if diagnostic_audit['training_window'] != TRAIN_WINDOW or diagnostic_audit['validation_window'] != VALIDATION_WINDOW:
        raise ValueError('Diagnostic split differs')
    targets,frequencies,auxiliary=load_targets(diagnostic,users,validation_support,diagnostic_audit)
    tie_targets={seed:{} for seed in SEEDS}
    for row in _rows(run/'tie-targets.csv',['seed','user_id','target_ids'],True):
        seed=int(row['seed']);user=row['user_id']
        if seed not in tie_targets or user not in frequencies or user in tie_targets[seed]:raise ValueError('Invalid saved tie resolution')
        tie_targets[seed][user]=tuple(row['target_ids'].split())
    for seed in SEEDS:
        check_deadline(deadline)
        if tie_targets[seed] != resolve_targets(frequencies,seed):raise ValueError('Saved shared tie resolution differs from frozen counts')
    reference_ties={}
    for row in _rows(run/'tie-sensitivity.csv',['seed','method','cohort','users','map_at_5']):
        key=(int(row['seed']),row['method'],row['cohort'])
        if key in reference_ties:raise ValueError('Duplicate saved sensitivity row')
        reference_ties[key]=float(row['map_at_5']) if row['map_at_5'] else None
    baseline=read_predictions(run/'validation-predictions.csv',users,validation_support)
    metrics,_=score_methods({'baseline':baseline},targets,users,auxiliary)
    saved_metrics=json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
    for row in metrics:
        if row['map_at_5'] is not None:_close(row['map_at_5'],saved_metrics['cohorts'][row['cohort']]['map_at_5'],'Saved baseline MAP')
    with np.load(run/'ppr-scores.npz',allow_pickle=False) as saved:
        score_users=tuple(saved['user_ids'].tolist());items=tuple(saved['item_ids'].tolist());scores=saved['scores'].copy()
    diagnostics=list(_rows(run/'ppr-diagnostics.csv',['user_id','status','fixed_point_residual','max_mass_error']))
    if [r['user_id'] for r in diagnostics]!=list(users):raise ValueError('Saved diagnostic cohort differs')
    expected_warm=tuple(r['user_id'] for r in diagnostics if r['status']=='converged')
    if items!=graph_items or score_users!=expected_warm or len(score_users)!=len(set(score_users)):
        raise ValueError('Saved baseline scores have different users/nodes')
    for row in diagnostics:
        if row['status']=='converged':
            for key in ('fixed_point_residual','max_mass_error'):
                value=float(row[key])
                if not math.isfinite(value) or not 0<=value<=1e-10+1e-14:raise ValueError('Saved numerical check failed')
        elif row['status']!='popularity_fallback':raise ValueError('Unknown saved PPR status')
    popularity=PopularityModel(counts,frozenset(score_users),source_audit['training_match_rows'])
    if scores.shape != (len(score_users),len(items)) or not np.isfinite(scores).all() or np.any(scores < 0) or np.max(np.abs(scores.sum(axis=1)-1),initial=0)>1e-10:
        raise ValueError('Saved baseline scores are not valid probabilities')
    val_ranker=FastRanker(items,popularity,validation_support);future_ranker=FastRanker(items,popularity,future)
    score_index={u:i for i,u in enumerate(score_users)}
    reranked={u:val_ranker.rank(scores[score_index[u]]) if u in score_index else val_ranker.fallback for u in users}
    if reranked != baseline:raise ValueError('Saved baseline scores do not reproduce validation predictions')
    future_predictions={u:future_ranker.rank(scores[score_index[u]]) if u in score_index else future_ranker.fallback for u in users}
    if future_predictions != read_predictions(prior/'submission.csv',users,future):raise ValueError('Saved baseline scores do not reproduce existing future CSV')
    context={'users':users,'future_candidates':sorted(future,key=int),'validation_candidates':sorted(validation_support,key=int),
             'targets':targets,'auxiliary_users':auxiliary,'baseline_configuration':'counts-d050',
             'training_window':TRAIN_WINDOW,'validation_window':VALIDATION_WINDOW,
             'prediction_window':{'start_inclusive':FUTURE_START.isoformat(),'end_exclusive':FUTURE_END.isoformat()},
             'inputs':[{'path':str(p),'sha256':h} for p,h in hashes.items()]}
    atomic_json(output/'context.json',context)
    return {'context':context,'tie_targets':tie_targets,'future_support':future_support,'validation_support':validation_support,
            'reference_ties':reference_ties,'hashes':hashes,'baseline':baseline,'baseline_future':future_predictions,
            'baseline_diagnostics':source_audit['convergence'],'val_ranker':val_ranker,'future_ranker':future_ranker}


def prepare_trials(prepared,exam,deadline):
    source=exam/'dataset-sweet-tv';matching=exam/'derived'/'matching'
    def schedule_rows():
        for n,row in enumerate(_rows(source/'export_arh_31-42-final.csv',['tv_show_id','start_time','duration'])):
            if n%4096==0:check_deadline(deadline)
            yield row
    future=candidate_programmes(schedule_rows(),FUTURE_START,FUTURE_END)
    if future!=prepared['future_support'].keys():raise ValueError('Future snapshot differs from schedule availability')
    high=Counter()
    def training_rows():
        for n,row in enumerate(_unique_training_rows(matching/'train-matches.csv')):
            if n%4096==0:check_deadline(deadline)
            if 5*int(row['overlap_seconds'])>=4*int(row['duration_seconds']):high[row['user_id'],row['tv_show_id']]+=1
            yield row
    data=build_interactions(training_rows())
    metadata=_training_metadata(matching/'data-quality.json')
    if metadata['audit']['matched_broadcasts']!=data.popularity.training_rows:raise ValueError('Original training count changed')
    graph=load_graph(exam/'derived'/'ppr-damping-050',data)
    for support in (prepared['future_support'],prepared['validation_support']):
        if any(data.popularity.counts.get(i,0)!=n for i,n in support.items()):raise ValueError('Training fallback differs')
    if data.item_ids!=prepared['val_ranker'].item_ids:raise ValueError('Training item order differs')
    prepared.update(data=data,strict=strict_profile(data,high),graph=graph)


def compute(prepared, damping, restart, deadline):
    data=prepared['data'] if restart=='all_counts' else prepared['strict']
    graph=prepared['graph'];users=prepared['context']['users'];index={u:i for i,u in enumerate(data.user_ids)}
    warm=[u for u in users if u in index]
    vr,fr=prepared['val_ranker'],prepared['future_ranker']
    predictions={u:vr.fallback for u in users if u not in index};future={u:fr.fallback for u in users if u not in index}
    diagnostics={'converged_users':len(warm),'cold_fallback_users':len(users)-len(warm),'max_iterations':0,'max_residual':0.,'max_mass_error':0.}
    config=PPRConfig(damping=damping)
    for offset in range(0,len(warm),32):
        check_deadline(deadline);batch=warm[offset:offset+32]
        seeds=np.zeros((len(graph.item_ids),len(batch)))
        for col,u in enumerate(batch):
            indices,counts=data.by_user[index[u]];seeds[indices,col]=counts/counts.sum()
        result=personalized_pagerank(graph.transition,seeds,config)
        diagnostics['max_iterations']=max(diagnostics['max_iterations'],int(result.iterations.max()))
        diagnostics['max_residual']=max(diagnostics['max_residual'],float(result.residual.max()))
        diagnostics['max_mass_error']=max(diagnostics['max_mass_error'],float(result.max_mass_error.max()))
        for col,u in enumerate(batch):
            predictions[u]=vr.rank(result.scores[:,col]);future[u]=fr.rank(result.scores[:,col])
    return predictions,future,diagnostics


def save_checkpoint(prepared,output,damping,restart,predictions,future,diagnostics,sequence,deadline,reused=False):
    check_deadline(deadline)
    context=prepared['context'];users=context['users'];aux=context['auxiliary_users']
    metrics,per_user=score_methods({'candidate':predictions},context['targets'],users,aux)
    tie_rows=[]
    for seed in SEEDS:
        check_deadline(deadline)
        scored,_=score_methods({'candidate':predictions},prepared['tie_targets'][seed],users,aux)
        for row in scored:
            count_reference=prepared['reference_ties'][seed,'ppr_counts_050',row['cohort']]
            uniform_reference=prepared['reference_ties'][seed,'ppr_050',row['cohort']]
            if reused and row['map_at_5'] is not None:_close(row['map_at_5'],count_reference,'Reused baseline shared-tie MAP')
            tie_rows.append({'seed':seed,**row,'delta_vs_counts_d050':row['map_at_5']-count_reference if row['map_at_5'] is not None else None,
                             'delta_vs_uniform_d050':row['map_at_5']-uniform_reference if row['map_at_5'] is not None else None})
    summaries=[]
    for cohort in ('all_users','at_least_five_coverage80'):
        rows=[r for r in tie_rows if r['cohort']==cohort];values=[r['map_at_5'] for r in rows if r['map_at_5'] is not None]
        summary={'cohort':cohort,'users':rows[0]['users'],'map_mean':statistics.mean(values) if values else None,
                 'map_sample_sd':statistics.stdev(values) if values else None,'map_sample_min':min(values) if values else None,'map_sample_max':max(values) if values else None}
        for comparator in ('counts_d050','uniform_d050'):
            deltas=[r['delta_vs_'+comparator] for r in rows if r['delta_vs_'+comparator] is not None]
            summary.update({f'delta_mean_vs_{comparator}':statistics.mean(deltas) if deltas else None,
                            f'delta_sample_min_vs_{comparator}':min(deltas) if deltas else None,
                            f'delta_sample_max_vs_{comparator}':max(deltas) if deltas else None,
                            f'wins_vs_{comparator}':sum(d>1e-12 for d in deltas) if deltas else None,
                            f'ties_vs_{comparator}':sum(abs(d)<=1e-12 for d in deltas) if deltas else None,
                            f'losses_vs_{comparator}':sum(d < -1e-12 for d in deltas) if deltas else None})
        summaries.append(summary)
    # Every promotion binds to the same original files. The supervisor can interrupt these checks.
    for p,h in prepared['hashes'].items():
        check_deadline(deadline)
        if file_hash(p)!=h:raise ValueError(f'Original input changed during trial: {p}')
    name=configuration_name(damping,restart);folder=output/'completed'/name;folder.mkdir(parents=True,exist_ok=False)
    for filename,values in [('validation-predictions.csv',predictions),('submission.csv',future)]:
        _save(folder/filename,['user_id','tv_show_id'],({'user_id':u,'tv_show_id':' '.join(values[u])} for u in users))
    validate_submission(folder/'submission.csv',users,context['future_candidates'])
    read_predictions(folder/'validation-predictions.csv',users,context['validation_candidates'])
    _save(folder/'per-user.csv',list(per_user[0]),per_user)
    _save(folder/'tie-sensitivity.csv',list(tie_rows[0]),tie_rows)
    atomic_json(folder/'tie-summary.json',{'interpretation':'Empirical sensitivity using the saved shared resolutions, not exact bounds or confidence intervals.',
                                          'completed_seeds':list(SEEDS),'results':summaries})
    atomic_json(folder/'diagnostics.json',diagnostics)
    result={'configuration':name,'damping':damping,'restart':restart,'parameters':asdict(PPRConfig(damping=damping)),
            'map_at_5':metrics[0]['map_at_5'],'auxiliary_map_at_5':metrics[1]['map_at_5'],
            'cohorts':{'all_users':len(users),'at_least_five_coverage80':len(aux)},'sequence':sequence,'reused_completed_result':reused,
            'verified':True,'csv_sha256':file_hash(folder/'submission.csv'),'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'artifacts':{p.name:file_hash(p) for p in folder.iterdir() if p.is_file()},
            'training_window':TRAIN_WINDOW,'graph_reused':True,'fallback':'Original training distinct-user popularity, including users with no strict qualifying history.',
            'source_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in
                              ('final_ppr_worker.py','final_ppr_experiment.py','ppr.py','evaluate_ppr_counts.py','evaluation.py','tie_sensitivity.py')}}
    check_deadline(deadline)
    verify_checkpoint(folder,context,result)
    check_deadline(deadline)
    # Final commit to selection.json occurs separately inside the timed worker.
    atomic_json(folder/'result.json',result)
    return result


def worker(exam,output,deadline):
    atomic_json(output/'worker-status.json',{'state':'preparing'})
    prepared=prepare(exam,output,deadline)
    baseline=save_checkpoint(prepared,output,.5,'all_counts',prepared['baseline'],prepared['baseline_future'],
                             prepared['baseline_diagnostics'],0,deadline,reused=True)
    committed=[baseline]
    publish_selection(output,committed,prepared['context'])
    atomic_json(output/'worker-status.json',{'state':'baseline_ready_preparing_trials'})
    prepare_trials(prepared,exam,deadline)
    sequence=0;attempts=[]
    def trial(damping,restart):
        nonlocal sequence
        check_deadline(deadline);sequence+=1
        attempt={'configuration':configuration_name(damping,restart),'state':'running'};attempts.append(attempt)
        atomic_json(output/'attempts.json',attempts)
        try:
            values=compute(prepared,damping,restart,deadline)
            result=save_checkpoint(prepared,output,damping,restart,*values,sequence,deadline)
            check_deadline(deadline)
            publish_selection(output,committed+[result],prepared['context'])
            committed.append(result)
            attempt['state']='completed';atomic_json(output/'attempts.json',attempts)
            return result
        except ValueError as error:
            attempt.update(state='failed',error=str(error));atomic_json(output/'attempts.json',attempts)
            return None
    results=run_stages(baseline,trial)
    atomic_json(output/'worker-status.json',{'state':'finished','completed_configurations':[r['configuration'] for r in results]})


def main():
    parser=argparse.ArgumentParser(description='Internal worker; invoke final_ppr_experiment.py instead.')
    parser.add_argument('--exam-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    try:
        worker(args.exam_dir,args.output_dir,args.deadline)
    except BudgetExpired:
        atomic_json(args.output_dir/'worker-status.json',{'state':'budget_expired'})
    except Exception as error:
        atomic_json(args.output_dir/'worker-status.json',{'state':'failed','error':str(error)})
        raise


if __name__=='__main__':main()
