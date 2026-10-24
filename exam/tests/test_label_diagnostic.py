"""Synthetic frozen-ranking diagnostics; never evaluate the real dataset here."""
from contextlib import redirect_stdout
import csv
from datetime import datetime, timedelta
import importlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from exam.evaluation import average_precision_at_5


LARGE = '18440717181607000261'
FIELDS = ['user_id', 'channel_id', 'tv_show_id', 'broadcast_start', 'broadcast_end', 'duration_seconds', 'overlap_seconds']


def match(user, item, overlap=80, duration=100, slot=0):
    begin = datetime(2020, 5, 18) + timedelta(minutes=slot*10)
    return dict(zip(FIELDS, [str(user), '1', str(item), begin.isoformat(),
                            (begin+timedelta(seconds=duration)).isoformat(), str(duration), str(overlap)]))


def save(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


class LabelDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.diagnose_labels'), 'Diagnostic not implemented')
        self.api = importlib.import_module('exam.diagnose_labels')

    def test_threshold_frequency_ties_and_known_answer_metrics(self):
        rows = [match(1, i, slot=n) for n, i in enumerate([1, 1, 2, 3, 4, 5, 6])]
        rows += [match(1, 7, 79, slot=9), match(2, 7, slot=10)]
        labels = self.api.build_labels(rows, ['1', '2', LARGE], {str(i) for i in range(1, 8)})
        self.assertEqual(labels.targets['current']['1'], frozenset(map(str, range(1, 8))))
        self.assertEqual(labels.targets['coverage80_all']['1'], frozenset(map(str, range(1, 7))))
        self.assertEqual(labels.targets['coverage80_top5']['1'], ('1','2','3','4','5'))
        self.assertEqual(labels.frequencies['1']['1'], 2)
        self.assertTrue(labels.profiles['1']['cutoff_tie'])
        self.assertEqual(labels.profiles['1']['cutoff_tie_size'], 5)
        self.assertEqual(labels.auxiliary_users, ('1',))
        predictions = {u: ('6','7','1','2','3') for u in ['1','2',LARGE]}
        comparison, per_user = self.api.score_predictions('test', predictions, labels)
        result = {(r['label_definition'],r['cohort']):r for r in comparison}
        self.assertAlmostEqual(result['current','all_users']['map_at_5'], .5)
        self.assertAlmostEqual(result['coverage80_all','all_users']['map_at_5'], ((1+2/3+3/4+4/5)/5+.5)/3)
        self.assertAlmostEqual(result['coverage80_top5','all_users']['map_at_5'], ((1/3+2/4+3/5)/5+.5)/3)
        self.assertAlmostEqual(result['coverage80_top5','at_least_five_coverage80']['map_at_5'], (1/3+2/4+3/5)/5)
        self.assertTrue(all(r['users']==1 for r in comparison if r['cohort']=='at_least_five_coverage80'))
        self.assertTrue(all(r['empty_target_users']==1 for r in comparison if r['cohort']=='all_users'))
        self.assertEqual(len(per_user),9)

    def test_odd_duration_and_clipped_airing_keep_full_duration(self):
        rows = [match(1, 1, 80, 101), match(1, 2, 81, 101, slot=1)]
        crossing = match(1,3,80,100,slot=2)
        crossing.update(broadcast_start='2020-05-17T23:59:40',broadcast_end='2020-05-18T00:01:20')
        rows.append(crossing)
        labels=self.api.build_labels(rows,['1'],{'1','2','3','4','5'})
        self.assertEqual(labels.targets['coverage80_all']['1'],frozenset({'2','3'}))
        comp,_=self.api.score_predictions('test',{'1':('1','2','3','4','5')},labels)
        self.assertTrue(all(r['map_at_5'] is None and r['users']==0 for r in comp if r['cohort']=='at_least_five_coverage80'))

    def test_filter_can_empty_a_target_without_dropping_the_user(self):
        labels=self.api.build_labels([match(1,1,79)],['1',LARGE],{'1','2','3','4','5'})
        table,_=self.api.score_predictions('test',{u:('1','2','3','4','5') for u in labels.users},labels)
        full={r['label_definition']:r for r in table if r['cohort']=='all_users'}
        self.assertEqual(full['current']['map_at_5'],.5)
        for name in ('coverage80_all','coverage80_top5'):
            self.assertEqual(full[name]['users'],2)
            self.assertEqual(full[name]['empty_target_users'],2)
            self.assertEqual(full[name]['map_at_5'],0)

    def test_matcher_accumulates_fragments_once_per_broadcast(self):
        from dataclasses import asdict
        from exam.matching import match_viewings
        views=[{'user_id':'1','vsetv_id':'1','start_time':'2020-05-18 00:00:00','stop_time':'2020-05-18 00:00:40','duraton':'40'},
               {'user_id':'1','vsetv_id':'1','start_time':'2020-05-18 00:01:00','stop_time':'2020-05-18 00:01:40','duraton':'40'}]
        schedule=[{'channel_id':'1','start_time':'18.05.2020 00:00:00','duration':'100','tv_show_id':'2'}]
        result=match_viewings(views+views[:1],schedule,window_start=datetime(2020,5,18),window_end=datetime(2020,7,27))
        row={k:(v.isoformat() if isinstance(v,datetime) else str(v)) for k,v in asdict(result.matches[0]).items()}
        labels=self.api.build_labels([row],['1'],{'2'})
        self.assertEqual(labels.frequencies['1']['2'],1)
        self.assertEqual(row['overlap_seconds'],'80')

    def test_duplicate_airing_bad_intervals_and_impossible_overlap_fail(self):
        good=match(1,1)
        for rows in ([good,good], [dict(good,overlap_seconds='101')], [dict(good,overlap_seconds='49')],
                     [dict(good,broadcast_start='2020-08-01T00:00:00',broadcast_end='2020-08-01T00:01:40')],
                     [dict(good,duration_seconds='101')], [dict(good,tv_show_id='0')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                self.api.build_labels(rows,['1'],{'1','2'})

    def fixture(self,root):
        source,matched,runs,output=(root/n for n in ('source','matched','runs','output'))
        source.mkdir();matched.mkdir();runs.mkdir()
        users=[str(i) for i in range(1,1260)]+[LARGE]
        save(source/'submission.csv',['user_id','tv_show_id'],[{'user_id':u,'tv_show_id':'0 0 0 0 0'} for u in users])
        matches=[match(1,i,slot=n) for n,i in enumerate([1,1,2,3,4,5,6])]+[match(1,7,79,slot=9),match(2,7,slot=10)]
        save(matched/'validation-matches.csv',FIELDS,matches)
        windows={'train':('2020-03-09T00:00:00','2020-05-18T00:00:00'),
                 'validation':('2020-05-18T00:00:00','2020-07-27T00:00:00')}
        metadata={'partitions':{n:{'window_start_inclusive':s,'window_end_exclusive':e} for n,(s,e) in windows.items()}}
        metadata['partitions']['validation']['audit']={'matched_broadcasts':len(matches)}
        (matched/'data-quality.json').write_text(json.dumps(metadata),encoding='utf-8')
        configs=[('popularity','distinct_training_user_popularity',None),
                 ('als-tuning/als-06','implicit_als',{'factors':32,'iterations':15,'alpha':1.,'regularization':10.,'seed':42,'confidence_weighting':'binary'}),
                 ('ppr','personalized_pagerank',{'damping':.85,'tolerance':1e-10,'max_iterations':200}),
                 ('ppr-damping-050','personalized_pagerank',{'damping':.5,'tolerance':1e-10,'max_iterations':200})]
        for folder,method,params in configs:
            d=runs/folder;d.mkdir(parents=True)
            per=[]
            for u in users:
                target=set(map(str,range(1,8))) if u=='1' else {'7'} if u=='2' else set()
                ranking='6 7 1 2 3'
                per.append({'user_id':u,'relevant_count':len(target),'relevant_ids':' '.join(sorted(target,key=int)),
                            'tv_show_id':ranking,'hits_at_5':len(target&set(ranking.split())),
                            'ap_at_5':average_precision_at_5(target,ranking.split())})
            save(d/'per-user.csv',list(per[0]),per)
            save(d/'validation-predictions.csv',['user_id','tv_show_id'],[{'user_id':u,'tv_show_id':'6 7 1 2 3'} for u in users])
            save(d/'candidate-popularity.csv',['tv_show_id','training_distinct_users'],[{'tv_show_id':str(i),'training_distinct_users':1} for i in range(1,8)])
            stats={'method':method,'metric':'MAP@5','candidate_programmes':7,'validation_match_rows':len(matches),
                   'validation_match_rows_in_cohort':len(matches),'relevant_user_item_pairs':8,
                   'cohorts':{'all_users':{'users':1260,'empty_target_users':1258,'map_at_5':1.5/1260}},
                   **{('training_window' if n=='train' else 'validation_window'):{'start_inclusive':s,'end_exclusive':e} for n,(s,e) in windows.items()}}
            if params is not None:stats['parameters']=params
            (d/'evaluation.json').write_text(json.dumps(stats),encoding='utf-8')
        return source,matched,runs,output

    def test_runner_is_separate_preserves_inputs_and_reproduces_old_scores(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            source,matched,runs,output=self.fixture(Path(t))
            before={p:p.read_bytes() for p in Path(t).rglob('*') if p.is_file()}
            with redirect_stdout(StringIO()):audit=self.api.run_diagnostic(source,matched,runs,output)
            self.assertTrue(audit['current_labels_reproduce_saved_results'])
            self.assertEqual(audit['cohorts'],{'all_users':1260,'at_least_five_coverage80':1})
            with (output/'comparison.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),24)
            self.assertEqual({r['method'] for r in table},{'popularity','als_tuned','ppr_085','ppr_050'})
            for p,b in before.items():self.assertEqual(p.read_bytes(),b)
            with self.assertRaises(FileExistsError):self.api.run_diagnostic(source,matched,runs,output)

    def test_mismatched_saved_results_pool_split_and_predictions_fail_before_writing(self):
        for mode in ('metric','pool','window','prediction','relevance'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
                source,matched,runs,output=self.fixture(Path(t));d=runs/'ppr'
                if mode in ('metric','window'):
                    p=d/'evaluation.json';data=json.loads(p.read_text())
                    if mode=='metric':data['cohorts']['all_users']['map_at_5']=.9
                    else:data['training_window']['end_exclusive']='2020-07-27T00:00:00'
                    p.write_text(json.dumps(data),encoding='utf-8')
                elif mode=='pool':
                    p=d/'candidate-popularity.csv';p.write_text(p.read_text().replace('7,1','8,1'),encoding='utf-8')
                elif mode=='prediction':
                    p=d/'validation-predictions.csv';p.write_text(p.read_text().replace('6 7 1 2 3','6 6 1 2 3',1),encoding='utf-8')
                else:
                    p=d/'per-user.csv';p.write_text(p.read_text().replace('1 2 3 4 5 6 7','1 2 3 4 5 6 8',1),encoding='utf-8')
                with redirect_stdout(StringIO()),self.assertRaises(ValueError):self.api.run_diagnostic(source,matched,runs,output)
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
