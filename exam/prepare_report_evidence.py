"""Refresh two report examples from frozen outputs; no model fitting."""
import csv, json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.sparse import load_npz
from ppr import personalized_pagerank, PPRConfig
from evaluation import average_precision_at_5

E=Path(__file__).resolve().parent
out=E/'derived/report';out.mkdir(exist_ok=True)
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: yield from csv.DictReader(f)
users=json.loads((E/'derived/business-analysis/selected-users.json').read_text(encoding='utf-8'))['users']
ids=[u['user_id'] for u in users]
old=list(rows(E/'derived/business-analysis/recommendations.csv'))
labels={r['user_id']:r['top5_ids'].split() for r in rows(E/'derived/label-diagnostic/labels.csv') if r['user_id'] in ids}
pred={r['user_id']:r['tv_show_id'].split() for r in rows(E/'derived/ppr-final-8min/completed/coverage80-d025/validation-predictions.csv') if r['user_id'] in ids}
items=[r['tv_show_id'] for r in rows(E/'derived/ppr-damping-050/graph-items.csv')];index={i:j for j,i in enumerate(items)}
counts={u:Counter() for u in ids};allcounts={u:Counter() for u in ids}
for r in rows(E/'derived/matching/train-matches.csv'):
    u=r['user_id']
    if u in counts:
        allcounts[u][r['tv_show_id']]+=1
        if 5*int(r['overlap_seconds'])>=4*int(r['duration_seconds']):counts[u][r['tv_show_id']]+=1
seeds=np.zeros((len(items),2))
for k,u in enumerate(ids):
    for i,n in counts[u].items():seeds[index[i],k]=n
scores=personalized_pagerank(load_npz(E/'derived/ppr-damping-050/transition.npz'),seeds,PPRConfig(damping=.25)).scores
need=set(i for p in pred.values() for i in p);meta={i:Counter() for i in need}
for r in rows(E/'dataset-sweet-tv/export_arh_11-20-final.csv'):
    i=str(int(r['tv_show_id']))
    if i in meta:meta[i][(r['tv_show_title'],r['tv_show_category'])]+=1
report=[]
for k,u in enumerate(ids):
    rr=[dict(r) for r in old if r['user_id']==u and r['method'] in ['popularity','tuned_als']]
    for rank,i in enumerate(pred[u],1):
        title,cat=sorted(meta[i],key=lambda x:(-meta[i][x],x))[0]
        rr.append(dict(user_id=u,method='selected_ppr',rank=rank,programme_id=i,title=title,category=cat,score=float(scores[index[i],k]),training_broadcasts=allcounts[u][i]))
    for r in rr:r['validation_hit']=r['programme_id'] in labels[u]
    ap={m:average_precision_at_5(labels[u],[r['programme_id'] for r in rr if r['method']==m]) for m in ['popularity','tuned_als','selected_ppr']}
    report.append(dict(history=users[k],targets=labels[u],strict_training_counts=counts[u],recommendations=rr,ap=ap))
(out/'two-users.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{'user':x['history']['user_id'],'targets':x['targets'],'ap':x['ap']} for x in report],indent=2))
