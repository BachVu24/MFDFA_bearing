"""Focused checks for Step9 lock enforcement and evidence accounting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('feature_lock', HERE/'load_locked_features.py')
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)
OUT = HERE.parent

class FeatureLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(lock.REPOSITORY/'result/test/step7/features/analysis_representatives.csv')

    def test_fixed_columns_and_qc_exclusion(self):
        meta, X, excluded = lock.locked_inputs(self.frame)
        self.assertEqual(list(X.columns), ['delta_alpha','alpha0'])
        self.assertEqual((len(X),len(excluded)),(48,112))
        self.assertEqual(meta.recording.nunique(),len(X))
        self.assertTrue(np.isfinite(X).all().all())
        self.assertTrue(meta.analysis_valid.all())

    def test_both_channels_use_same_features_without_extra_samples(self):
        meta, X, excluded = lock.locked_inputs(self.frame,'input_voltage')
        self.assertEqual(list(X.columns),['delta_alpha','alpha0'])
        self.assertEqual((len(X),len(excluded)),(76,84))
        self.assertTrue(meta.recording.str.startswith('spur_').all())

    def test_duplicate_recording_rejected(self):
        row = self.frame[(self.frame.pipeline=='raw')&(self.frame.channel=='output_voltage')].iloc[[0]]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            lock.locked_inputs(pd.concat([self.frame,row]))

    def test_string_false_cannot_bypass_qc(self):
        frame=self.frame.copy();frame['analysis_valid']=frame.analysis_valid.astype(str)
        with self.assertRaisesRegex(ValueError,'explicit booleans'):
            lock.locked_inputs(frame)

    def test_nonfinite_valid_feature_excluded(self):
        frame=self.frame.copy();index=frame[(frame.pipeline=='raw')&(frame.channel=='output_voltage')&frame.analysis_valid].index[0]
        frame.loc[index,'alpha0']=np.nan
        meta,X,excluded=lock.locked_inputs(frame)
        self.assertEqual((len(X),len(excluded)),(47,113))

    def test_config_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            config=json.loads((lock.LOCK_DIRECTORY/'locked_feature_set.json').read_text())
            config['features'].append('delta_h')
            (directory/'locked_feature_set.json').write_text(json.dumps(config))
            (directory/'locked_feature_set.sha256').write_text((lock.LOCK_DIRECTORY/'locked_feature_set.sha256').read_text())
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                lock.load_locked_config(directory)
            (directory/'locked_feature_set.sha256').write_text(lock.digest(directory/'locked_feature_set.json'))
            with self.assertRaisesRegex(ValueError,'policy changed'):
                lock.load_locked_config(directory)

    def test_coverage_and_complementarity_accounting(self):
        summary=pd.read_csv(OUT/'feature_summary.csv')
        self.assertEqual(len(summary),96)
        self.assertTrue((summary.disjoint_pair_conditions<=summary.evaluable_pair_conditions).all())
        self.assertTrue((summary.evaluable_pair_conditions<=summary.expected_pair_conditions).all())
        np.testing.assert_allclose(summary.qc_coverage,summary.valid_signals/summary.expected_signals,rtol=1e-14,atol=1e-14)
        union=pd.read_csv(OUT/'tables/feature_set_separation.csv')
        for keys,g in union[union.representation=='RAW'].groupby(['dataset','channel']):
            counts=g.set_index('feature_set').disjoint_on_at_least_one_feature
            extra=counts['delta_alpha+alpha0+delta_h']-counts['delta_alpha+alpha0']
            expected=0 if keys==('Helical','output_voltage') else 1
            self.assertEqual(extra,expected)
        spur=summary[(summary.dataset=='Spur')&(summary.representation=='RAW')]
        self.assertTrue(spur[['fault_partial_eta_squared','speed_partial_eta_squared','load_partial_eta_squared']].isna().all().all())
        helical=summary[(summary.dataset=='Helical')&(summary.representation=='RAW')]
        self.assertTrue(helical.fault_partial_eta_squared.notna().all())

if __name__ == '__main__':
    unittest.main(verbosity=2)
