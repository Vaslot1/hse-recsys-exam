"""Final prediction checks using tiny synthetic inputs; no real Kaggle CSV generated."""
from contextlib import redirect_stdout
import csv
import importlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest


LARGE_ID = '18440717181607000261'


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class KaggleTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.generate_kaggle'), 'Final generator not implemented')
        self.api = importlib.import_module('exam.generate_kaggle')

    def fixture(self, root):
        source, matched, output = root/'source', root/'matched', root/'output'
        source.mkdir(); matched.mkdir()
        users = [str(i) for i in range(1, 1260)] + [LARGE_ID]
        write_csv(source/'submission.csv', ['user_id', 'tv_show_id'], [(u, '0 0 0 0 0') for u in users])
        write_csv(source/'export_arh_31-42-final.csv', ['start_time', 'duration', 'tv_show_id'],
                  [('26.07.2020 23:59:00', 120, 2)] +
                  [('27.07.2020 00:00:00', 600, i) for i in [4, 5, 6, 99, 0]] +
                  [('18.10.2020 23:59:00', 60, 99), ('19.10.2020 00:00:00', 600, 100),
                   ('26.07.2020 23:59:00', 60, 101), ('27.07.2020 00:00:00', 0, 102)])
        header = ['user_id', 'channel_id', 'tv_show_id', 'broadcast_start', 'broadcast_end', 'duration_seconds', 'overlap_seconds']
        write_csv(matched/'train-matches.csv', header,
                  [(u, 1, item, '2020-03-09 12:00:00', '2020-03-09 12:10:00', 600, 300)
                   for u,item in [(LARGE_ID, 2), (LARGE_ID, 10), (2, 3), (2, 10)]])
        metadata = {'partitions': {'train': {'window_start_inclusive': '2020-03-09T00:00:00',
                                             'window_end_exclusive': '2020-05-18T00:00:00',
                                             'audit': {'matched_broadcasts': 4}}}}
        (matched/'data-quality.json').write_text(json.dumps(metadata), encoding='utf-8')
        return source, matched, output, users

    def test_generator_needs_only_training_and_future_inputs_preserves_ids_and_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source, matched, output, users = self.fixture(root)
            before = {p: p.read_bytes() for folder in (source, matched) for p in folder.iterdir()}
            with redirect_stdout(StringIO()):
                audit = self.api.run_generation(source, matched, output)
            self.assertEqual(audit['parameters']['damping'], 0.5)
            self.assertEqual(audit['training_users'], 2)
            self.assertEqual(audit['graph']['nodes'], 3)
            self.assertEqual(audit['candidate_programmes'], 5)
            self.assertEqual(audit['cold_candidates'], 4)
            self.assertEqual(audit['csv_validation']['users'], 1260)
            self.assertEqual(audit['convergence']['cold_fallback_users'], 1258)
            self.assertNotIn('map_at_5', audit)
            with (output/'submission.csv').open(newline='', encoding='utf-8') as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual([r['user_id'] for r in predictions], users)
            self.assertEqual(predictions[-1]['user_id'], LARGE_ID)
            self.assertTrue(all(r['tv_show_id'] == '2 4 5 6 99' for r in predictions))
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual(audit['csv_sha256'], self.api.sha256(output/'submission.csv'))
            self.assertFalse((output/'submission.csv.tmp').exists())
            with self.assertRaises(FileExistsError):
                self.api.run_generation(source, matched, output)
            # Even an invalid validation file cannot affect generation; it must not be read.
            (matched/'validation-matches.csv').write_text('INVALID VALIDATION INPUT', encoding='utf-8')
            other = root/'other'
            with redirect_stdout(StringIO()):
                self.api.run_generation(source, matched, other)
            self.assertEqual((output/'submission.csv').read_bytes(), (other/'submission.csv').read_bytes())

    def test_submission_validator_rejects_user_and_recommendation_errors(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/'submission.csv'
            users = [str(i) for i in range(1,1260)]+[LARGE_ID]
            good = [[u, '2 4 5 6 99'] for u in users]
            cases = []
            for value in ['2 2 5 6 99', '0 4 5 6 99', '2 4 5 6 100', '2 4 5 6',
                          '2 4 5 6 99 100', '2  4 5 6 99', '2.0 4 5 6 99', '02 4 5 6 99']:
                rows = [r[:] for r in good]; rows[0][1] = value; cases.append((['user_id','tv_show_id'],rows))
            for changed in ['2', '99999', '1.0']:
                rows = [r[:] for r in good]; rows[0][0] = changed; cases.append((['user_id','tv_show_id'],rows))
            cases += [(['user_id','tv_show_id'],good[:-1]), (['user_id','tv_show_id'],good+[good[0]]),
                      (['user_id','tv_show_id'],[good[1],good[0]]+good[2:]),
                      (['index','user_id','tv_show_id'],good)]
            for header,rows in cases:
                with self.subTest(first=rows[0], size=len(rows), header=header):
                    write_csv(path,header,rows)
                    with self.assertRaises(ValueError):
                        self.api.validate_submission(path,users,{'2','4','5','6','99'})
            write_csv(path,['user_id','tv_show_id'],good)
            self.assertEqual(self.api.validate_submission(path,users,{'2','4','5','6','99'})['users'],1260)

    def test_bad_train_window_or_contamination_fails_before_outputs(self):
        for mode in ('metadata','out_of_window','excess_overlap','row_count'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                source,matched,output,_=self.fixture(Path(directory))
                if mode in ('metadata','row_count'):
                    path=matched/'data-quality.json'; metadata=json.loads(path.read_text())
                    if mode=='metadata':metadata['partitions']['train']['window_end_exclusive']='2020-07-27T00:00:00'
                    else:metadata['partitions']['train']['audit']['matched_broadcasts']=5
                    path.write_text(json.dumps(metadata),encoding='utf-8')
                else:
                    path=matched/'train-matches.csv'; text=path.read_text()
                    if mode=='out_of_window':text=text.replace('2020-03-09','2020-06-01')
                    else:text=text.replace(',600,300',',600,900')
                    path.write_text(text,encoding='utf-8')
                with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                    self.api.run_generation(source,matched,output)
                self.assertFalse(output.exists())

    def test_template_count_duplicates_and_small_candidate_pool_fail(self):
        for mode in ('missing_user','duplicate_user','small_pool'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                source,matched,output,users=self.fixture(Path(directory))
                if mode=='small_pool':
                    write_csv(source/'export_arh_31-42-final.csv',['start_time','duration','tv_show_id'],
                              [('27.07.2020 00:00:00',600,i) for i in [2,4,5,6]])
                else:
                    users=users[:-1] if mode=='missing_user' else users[:-1]+['1']
                    write_csv(source/'submission.csv',['user_id','tv_show_id'],[(u,'0 0 0 0 0') for u in users])
                with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                    self.api.run_generation(source,matched,output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
