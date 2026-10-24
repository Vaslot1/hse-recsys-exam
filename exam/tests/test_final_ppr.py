"""Synthetic tests for the budgeted final experiment. No real dataset execution."""
import csv
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import importlib.util
import json
from io import StringIO
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from exam.als import build_interactions
from exam.ppr import build_graph, rank_scores


class FinalPPRTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.final_ppr_worker'))
        self.api=importlib.import_module('exam.final_ppr_worker')
        self.supervisor=importlib.import_module('exam.final_ppr_experiment')

    def test_fast_ranker_matches_existing_ties_unavailable_and_cold_completion(self):
        data=build_interactions([{'user_id':u,'tv_show_id':i} for u,i in [('1','2'),('1','3'),('2','3'),('2','10')]])
        graph=build_graph(data)
        for pool in ({'2','3','10','4','5','6'},{'3','4','5','6','7'}):
            ranker=self.api.FastRanker(graph.item_ids,data.popularity,pool)
            for scores in ([0,0,0],[.1,.1,.8],[.2,.7,.1]):
                self.assertEqual(ranker.rank(scores),rank_scores(graph,scores,data.popularity,pool))
            self.assertEqual(ranker.fallback,data.popularity.top_five(pool))

    def test_strict_restart_keeps_graph_items_and_original_popularity(self):
        data=build_interactions([{'user_id':u,'tv_show_id':i} for u,i in [('1','2'),('1','2'),('1','3'),('2','3')]])
        strict=self.api.strict_profile(data,{('1','3'):1})
        self.assertEqual(strict.user_ids,('1',))
        self.assertEqual(strict.item_ids,data.item_ids)
        self.assertIs(strict.popularity,data.popularity)
        np.testing.assert_array_equal(strict.by_user[0][0],[1])
        np.testing.assert_array_equal(strict.by_user[0][1],[1.])

    def test_stages_use_completed_winner_and_do_not_select_on_sensitivity(self):
        calls=[]
        baseline={'configuration':'counts-d050','damping':.5,'restart':'all_counts','map_at_5':.3}
        scores={.05:.2,.1:.4,.25:.35}
        def trial(damping,restart):
            calls.append((damping,restart))
            return {'configuration':str(calls[-1]),'damping':damping,'restart':restart,
                    'map_at_5':scores[damping] if restart=='all_counts' else .38}
        results=self.api.run_stages(baseline,trial)
        self.assertEqual(calls,[(.05,'all_counts'),(.1,'all_counts'),(.25,'all_counts'),(.1,'coverage80_counts')])
        self.assertEqual(max(results,key=lambda r:r['map_at_5'])['damping'],.1)
        self.assertEqual(max(results,key=lambda r:r['map_at_5'])['restart'],'all_counts')

    def checkpoint(self,root,name,score):
        folder=root/'completed'/name;folder.mkdir(parents=True)
        users=[str(i) for i in range(1,1261)]
        context={'users':users,'future_candidates':['1','2','3','4','5'],'validation_candidates':['1','2','3','4','5'],
                 'targets':{u:['1','2','3','4','5'] for u in users},'auxiliary_users':users,
                 'baseline_configuration':'counts-d050'}
        # Known score is 1 for exact five targets; supervisor recomputes it.
        self.assertEqual(score,1.)
        self.supervisor.atomic_json(root/'context.json',context)
        for file in ['submission.csv','validation-predictions.csv']:
            with (folder/file).open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f);w.writerow(['user_id','tv_show_id']);w.writerows((u,'1 2 3 4 5') for u in users)
        self.supervisor.atomic_json(folder/'tie-summary.json',{'interpretation':'Empirical sensitivity, not bounds.','completed_seeds':list(range(20))})
        result={'configuration':name,'damping':.5,'restart':'all_counts','map_at_5':1.,'auxiliary_map_at_5':1.,
                'cohorts':{'all_users':1260,'at_least_five_coverage80':1260},'verified':True,
                'csv_sha256':self.supervisor.file_hash(folder/'submission.csv'),
                'artifacts':{n:self.supervisor.file_hash(folder/n) for n in ['submission.csv','validation-predictions.csv','tie-summary.json']}}
        self.supervisor.atomic_json(folder/'result.json',result)
        self.supervisor.publish_selection(root,[result],context)
        return folder

    def test_checkpoint_hash_and_metrics_are_verified_before_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);folder=self.checkpoint(root,'counts-d050',1.)
            result=self.supervisor.verify_checkpoint(folder,json.loads((root/'context.json').read_text()))
            self.assertEqual(result['map_at_5'],1.)
            with (folder/'submission.csv').open('a') as f:f.write('99999,1 2 3 4 5\n')
            with self.assertRaises(ValueError):self.supervisor.verify_checkpoint(folder,json.loads((root/'context.json').read_text()))

    def test_deadline_stops_worker_and_exports_only_completed_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            root=Path(t);self.checkpoint(root,'counts-d050',1.)
            partial=root/'completed'/'partial';partial.mkdir();(partial/'submission.csv').write_text('partial')
            start=time.monotonic()
            with patch.object(self.supervisor,'verify_checkpoint',side_effect=AssertionError('Supervisor must not do blocking verification')):
                result=self.supervisor.supervise([sys.executable,'-B','-c','import time; time.sleep(30)'],root,1.5)
            self.assertLess(time.monotonic()-start,3)
            self.assertTrue(result['budget_expired'])
            self.assertEqual(result['configuration'],'counts-d050')
            self.assertEqual(self.supervisor.file_hash(Path(result['submission_path'])),result['csv_sha256'])
            self.assertEqual((partial/'submission.csv').read_text(),'partial')

    def test_full_synthetic_sequence_reuses_graph_and_exports_separate_csv(self):
        from exam.tests import test_ppr_counts as count_tests
        from exam.evaluate_ppr_counts import run_experiment
        from exam.generate_kaggle import sha256
        from exam.tests.test_label_diagnostic import save
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            base=Path(t);exam=base/'exam';exam.mkdir()
            # Existing tiny fixture supplies all four comparator runs, graph and frozen diagnostic.
            source,matched,runs,diagnostic,counts=count_tests.CountRunnerTests().fixture(exam)
            (exam/'dataset-sweet-tv').mkdir();(exam/'derived').mkdir()
            # Relocate before rerunning diagnostic so all provenance paths identify this layout.
            import shutil
            for p in source.iterdir():shutil.copyfile(p,exam/'dataset-sweet-tv'/p.name)
            shutil.copytree(matched,exam/'derived'/'matching')
            for p in runs.iterdir():shutil.copytree(p,exam/'derived'/p.name)
            source=exam/'dataset-sweet-tv';matched=exam/'derived'/'matching';runs=exam/'derived'
            diagnostic=runs/'label-diagnostic';counts=runs/'ppr-count-restart-050'
            from exam.diagnose_labels import run_diagnostic
            with redirect_stdout(StringIO()):
                run_diagnostic(source,matched,runs,diagnostic)
                run_experiment(source,matched,runs,diagnostic,counts)
            save(source/'export_arh_31-42-final.csv',['tv_show_id','start_time','duration'],
                 [{'tv_show_id':str(i),'start_time':'27.07.2020 00:00:00','duration':100} for i in range(1,8)])
            prior=runs/'kaggle-ppr-count-restart-050';prior.mkdir()
            shutil.copyfile(counts/'validation-predictions.csv',prior/'submission.csv')
            shutil.copyfile(runs/'ppr-damping-050'/'candidate-popularity.csv',prior/'candidate-popularity.csv')
            self.supervisor.atomic_json(prior/'generation-audit.json',{'csv_sha256':sha256(prior/'submission.csv'),
                'restart_weighting':'watch_counts','parameters':{'damping':.5,'tolerance':1e-10,'max_iterations':200},
                'training_window':{'start_inclusive':'2020-03-09T00:00:00','end_exclusive':'2020-05-18T00:00:00'},
                'saved_validation_rankings_reproduced':True,'csv_validation':{'users':1260},
                'inputs':[{'path':str(source/'export_arh_31-42-final.csv'),'sha256':sha256(source/'export_arh_31-42-final.csv')}]})
            before={p:sha256(p) for p in exam.rglob('*') if p.is_file()}
            output=base/'final';output.mkdir()
            with redirect_stdout(StringIO()):self.api.worker(exam,output,time.monotonic()+90)
            self.assertEqual(json.loads((output/'worker-status.json').read_text())['state'],'finished')
            records=[json.loads(p.read_text()) for p in (output/'completed').glob('*/result.json')]
            self.assertEqual(len(records),5)
            self.assertEqual({r['damping'] for r in records if r['restart']=='all_counts'},{.05,.1,.25,.5})
            with redirect_stdout(StringIO()):selection=self.supervisor.supervise([sys.executable,'-B','-c','pass'],output,10)
            self.assertEqual(selection['map_at_5'],max(r['map_at_5'] for r in records))
            self.assertEqual(sha256(Path(selection['submission_path'])),selection['csv_sha256'])
            self.assertTrue(all(sha256(p)==h for p,h in before.items()))
            fallback=base/'fallback';fallback.mkdir()
            with patch.object(self.api,'prepare_trials',side_effect=self.api.BudgetExpired('synthetic preparation timeout')):
                with self.assertRaises(self.api.BudgetExpired):self.api.worker(exam,fallback,time.monotonic()+90)
            ready=json.loads((fallback/'selection.json').read_text())
            self.assertEqual(ready['configuration'],'counts-d050')
            self.assertEqual(sha256(Path(ready['submission_path'])),ready['csv_sha256'])


if __name__=='__main__':unittest.main()
