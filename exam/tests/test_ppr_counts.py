"""Count-restart and tie-resolution checks on synthetic data only."""
from contextlib import redirect_stdout
import csv
from dataclasses import asdict
import importlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from exam.als import build_interactions
from exam.ppr import build_graph, PPRConfig, predict_users
from exam.tests import test_label_diagnostic as label_tests
from exam.tests.test_label_diagnostic import FIELDS, match, save


class CountRestartTests(unittest.TestCase):
    def test_count_weighted_restart_matches_two_node_solution_and_keeps_default(self):
        data=build_interactions([{'user_id':'1','tv_show_id':i} for i in ['2','2','2','3']])
        graph=build_graph(data); config=PPRConfig(damping=.5)
        before=graph.weights.copy();transition=graph.transition.copy()
        count=predict_users(data,graph,['1','99'],['2','3','4','5','6'],config,restart_weighting='watch_counts')
        uniform=predict_users(data,graph,['1','99'],['2','3','4','5','6'],config)
        np.testing.assert_allclose(count.scores[0],[7/12,5/12],atol=1e-10,rtol=0)
        np.testing.assert_allclose(uniform.scores[0],[.5,.5],atol=1e-14)
        self.assertEqual(count.predictions['99'],uniform.predictions['99'])
        self.assertEqual((graph.weights!=before).nnz,0)
        self.assertEqual((graph.transition!=transition).nnz,0)
        self.assertLessEqual(count.diagnostics['1']['max_mass_error'],1e-10)
        with self.assertRaises(ValueError):
            predict_users(data,graph,['1'],['2','3','4','5','6'],config,restart_weighting='unknown')

    def test_count_restart_can_change_ranking_without_changing_popularity_ties(self):
        data=build_interactions([{'user_id':'1','tv_show_id':i} for i in ['2','3','3','3']])
        graph=build_graph(data);pool=['2','3','4','5','6'];config=PPRConfig(damping=.5)
        uniform=predict_users(data,graph,['1'],pool,config)
        counts=predict_users(data,graph,['1'],pool,config,restart_weighting='watch_counts')
        self.assertEqual(uniform.predictions['1'][:2],('2','3'))
        self.assertEqual(counts.predictions['1'][:2],('3','2'))
        np.testing.assert_allclose(counts.scores[0],[5/12,7/12],atol=1e-10,rtol=0)


class TieSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.tie_sensitivity'),'Tie sensitivity missing')
        self.api=importlib.import_module('exam.tie_sensitivity')

    def test_seeded_targets_respect_frequency_cutoff_and_are_reproducible(self):
        frequencies={'1':{'1':5,'2':4,'3':3,'4':2,'5':1,'6':1,'7':1},
                     '18440717181607000261':{},'3':{'99':2}}
        realizations=[]
        for seed in range(20):
            targets=self.api.resolve_targets(frequencies,seed)
            self.assertEqual(targets,self.api.resolve_targets(dict(reversed(list(frequencies.items()))),seed))
            self.assertEqual(set(targets['1'])&{'1','2','3','4'},{'1','2','3','4'})
            self.assertEqual(len(targets['1']),5)
            self.assertEqual(targets['3'],('99',))
            self.assertEqual(targets['18440717181607000261'],())
            realizations.append(targets['1'])
        self.assertGreater(len(set(realizations)),1)
        self.assertEqual(tuple(self.api.SEEDS),tuple(range(20)))

    def test_same_targets_for_all_methods_and_paired_summary(self):
        users=('1','2','3')
        frequencies={'1':{'1':5,'2':4,'3':3,'4':2,'5':1,'6':1},'2':{},'3':{'1':2}}
        predictions={'ppr_050':{u:('1','2','3','4','5') for u in users},
                     'ppr_counts_050':{u:('1','2','3','4','5') for u in users}}
        analysis=self.api.analyze(predictions,frequencies,users)
        self.assertEqual(len(analysis['per_seed']),80)
        self.assertEqual(len(analysis['target_rows']),60)
        for row in analysis['summary']:
            self.assertEqual(row['paired_delta_mean_vs_ppr_050'],0)
            self.assertEqual(row['paired_wins_vs_ppr_050'],0)
            self.assertEqual(row['paired_ties_vs_ppr_050'],20)
            self.assertEqual(row['users'],3 if row['cohort']=='all_users' else 1)
        no_aux=self.api.analyze({k:{'2':v['2']} for k,v in predictions.items()},{'2':{}},('2',))
        for row in no_aux['summary']:
            if row['cohort']=='at_least_five_coverage80':self.assertIsNone(row['map_mean'])
            else:self.assertEqual(row['map_mean'],0)

    def test_sampling_does_not_depend_on_method_order_or_predictions(self):
        from exam.evaluation import average_precision_at_5
        frequencies={'1':{str(i):1 for i in range(1,9)}}
        predictions={'ppr_050':{'1':('1','2','3','4','5')},'other':{'1':('8','7','6','5','4')}}
        analysis=self.api.analyze(predictions,frequencies,('1',))
        changed=self.api.analyze({'other':{'1':('1','3','5','7','8')},'ppr_050':predictions['ppr_050']},frequencies,('1',))
        self.assertEqual(analysis['target_rows'],changed['target_rows'])
        for row in analysis['per_seed']:
            target=analysis['target_rows'][row['seed']]['target_ids'].split()
            self.assertEqual(row['map_at_5'],average_precision_at_5(target,predictions[row['method']]['1']))


class CountRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.evaluate_ppr_counts'),'Count runner missing')
        self.api=importlib.import_module('exam.evaluate_ppr_counts')

    def fixture(self,root):
        from exam.diagnose_labels import run_diagnostic
        source,matched,runs,diagnostic=label_tests.LabelDiagnosticTests().fixture(root)
        training=[dict(match(1, i, slot=n),broadcast_start=f'2020-03-09T00:{n*10:02d}:00',
                       broadcast_end=f'2020-03-09T00:{n*10:02d}:50',duration_seconds='50',overlap_seconds='40')
                  for n,i in enumerate(['1','1','1','2'])]
        save(matched/'train-matches.csv',FIELDS,training)
        metadata=json.loads((matched/'data-quality.json').read_text())
        metadata['partitions']['train']['audit']={'matched_broadcasts':4}
        (matched/'data-quality.json').write_text(json.dumps(metadata),encoding='utf-8')
        data=build_interactions(training);graph=build_graph(data)
        # All saved candidate files must describe this one training snapshot.
        for folder in ['popularity','als-tuning/als-06','ppr','ppr-damping-050']:
            save(runs/folder/'candidate-popularity.csv',['tv_show_id','training_distinct_users'],
                 [{'tv_show_id':str(i),'training_distinct_users':int(i in (1,2))} for i in range(1,8)])
        g=runs/'ppr-damping-050'
        sparse.save_npz(g/'graph.npz',graph.weights);sparse.save_npz(g/'transition.npz',graph.transition)
        save(g/'graph-items.csv',['node_index','tv_show_id','training_distinct_users','weighted_degree','dangling'],
             [{'node_index':i,'tv_show_id':item,'training_distinct_users':1,'weighted_degree':1,'dangling':False} for i,item in enumerate(graph.item_ids)])
        (g/'graph.json').write_text(json.dumps({**graph.diagnostics,'parameters':asdict(PPRConfig(damping=.5)),
                                               'training_window':{'start_inclusive':'2020-03-09T00:00:00','end_exclusive':'2020-05-18T00:00:00'},
                                               'training_users':1,'training_match_rows':4}),encoding='utf-8')
        with redirect_stdout(StringIO()):run_diagnostic(source,matched,runs,diagnostic)
        return source,matched,runs,diagnostic,root/'counts'

    def test_synthetic_runner_preserves_inputs_graph_and_frozen_targets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
            source,matched,runs,diagnostic,output=self.fixture(Path(t))
            before={p:p.read_bytes() for p in Path(t).rglob('*') if p.is_file()}
            with redirect_stdout(StringIO()):audit=self.api.run_experiment(source,matched,runs,diagnostic,output)
            self.assertEqual(audit['parameters']['damping'],.5)
            self.assertEqual(audit['restart_weighting'],'watch_counts')
            self.assertEqual(audit['seeds'],list(range(20)))
            self.assertTrue(audit['saved_graph_reused'])
            self.assertTrue(audit['frozen_comparator_scores_reproduced'])
            with (output/'comparison.csv').open(newline='',encoding='utf-8') as f:comparison=list(csv.DictReader(f))
            self.assertEqual(len(comparison),10)
            with (output/'tie-sensitivity.csv').open(newline='',encoding='utf-8') as f:sensitivity=list(csv.DictReader(f))
            self.assertEqual(len(sensitivity),200)
            with np.load(output/'ppr-scores.npz',allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['scores'][0],[7/12,5/12],atol=1e-10)
            for p,b in before.items():self.assertEqual(p.read_bytes(),b)
            with self.assertRaises(FileExistsError):self.api.run_experiment(source,matched,runs,diagnostic,output)

    def test_modified_frozen_target_graph_or_old_input_fails_before_output(self):
        for mode in ('target','graph','old_input'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as t:
                source,matched,runs,diagnostic,output=self.fixture(Path(t))
                if mode=='target':
                    p=diagnostic/'labels.csv';p.write_text(p.read_text().replace('1 2 3 4 5\n','1 2 3 4 6\n',1),encoding='utf-8')
                elif mode=='graph':sparse.save_npz(runs/'ppr-damping-050'/'transition.npz',sparse.eye(2,format='csr'))
                else:
                    p=runs/'popularity'/'validation-predictions.csv';p.write_text(p.read_text().replace('6 7 1 2 3','7 6 1 2 3',1),encoding='utf-8')
                with redirect_stdout(StringIO()),self.assertRaises(ValueError):
                    self.api.run_experiment(source,matched,runs,diagnostic,output)
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
