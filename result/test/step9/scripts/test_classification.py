"""Check empirical leakage, fixed features, metrics, and missing-class handling."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

OUT=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('step9_runner',OUT/'scripts/run_classification.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=pd.read_csv(OUT/'classification_results.csv')
        cls.pred=pd.read_csv(OUT/'tables/out_of_fold_predictions.csv')
        cls.audit=pd.read_csv(OUT/'tables/fit_leakage_audit.csv')
        cls.frames={dataset:pd.read_csv(OUT/f'tables/{dataset.lower()}_eligible_samples.csv') for dataset in ['Helical','Spur']}

    def test_feature_lock_and_parameters(self):
        protocol=json.loads((OUT/'config/evaluation_protocol.json').read_text())
        self.assertEqual(protocol['features'],['delta_alpha','alpha0'])
        self.assertFalse(protocol['hyperparameter_search'])
        self.assertEqual(protocol['step8_config_sha256'],runner.digest(ROOT/'result/test/step8/config/locked_feature_set.json'))
        self.assertEqual(len(self.result),16)
        self.assertEqual(set(self.result.classifier),{'KNN','SVM','LDA','Random Forest'})

    def test_recording_disjoint_and_common_splits(self):
        for row in self.audit.itertuples():
            train=set(row.train_recordings.split(';'));test=set(row.test_recordings.split(';'))
            self.assertFalse(train&test)
            group=self.frames[row.dataset];group=group[group.channel.eq(row.channel)]
            self.assertEqual(train,set(group.loc[group.fold.ne(row.fold),'recording']))
            self.assertEqual(test,set(group.loc[group.fold.eq(row.fold),'recording']))
        self.assertTrue(self.pred.groupby(['dataset','channel','classifier','recording']).size().eq(1).all())
        self.assertTrue(self.pred.groupby(['dataset','recording']).fold.nunique().eq(1).all())
        for keys,g in self.pred.groupby(['dataset','channel']):
            self.assertTrue(g.groupby('recording').classifier.nunique().eq(4).all())

    def test_scaler_means_from_train_only(self):
        for row in self.audit.itertuples():
            if row.classifier=='Random Forest':
                self.assertEqual(row.scaling,'none required');continue
            frame=self.frames[row.dataset]
            train=frame[frame.channel.eq(row.channel)&frame.fold.ne(row.fold)]
            actual=np.array([row.scaler_mean_delta_alpha,row.scaler_mean_alpha0])
            np.testing.assert_allclose(actual,train[['delta_alpha','alpha0']].mean().to_numpy(),rtol=1e-12)

    def test_metrics_and_confusion_are_oof(self):
        pc=pd.read_csv(OUT/'per_class_metrics.csv')
        for row in self.result.itertuples():
            group=self.pred[self.pred.dataset.eq(row.dataset)&self.pred.channel.eq(row.channel)&self.pred.classifier.eq(row.classifier)]
            self.assertEqual(len(group),row.n_samples)
            self.assertAlmostEqual(row.accuracy,accuracy_score(group.state,group.predicted_state))
            self.assertAlmostEqual(row.macro_f1,f1_score(group.state,group.predicted_state,average='macro',zero_division=0))
            path=OUT/f'confusion_matrices/{row.dataset.lower()}_{row.channel}_{runner.MODULES[row.classifier]}.csv'
            cm=pd.read_csv(path).set_index('true_state')
            np.testing.assert_array_equal(cm.to_numpy(),confusion_matrix(group.state,group.predicted_state,labels=cm.index.tolist()))
        missing=pc[pc.support.eq(0)]
        self.assertTrue(missing[['precision','recall','f1']].isna().all().all())
        self.assertTrue(missing.status.str.startswith('NOT EVALUABLE').all())

    def test_split_allocation_does_not_use_features(self):
        for dataset,frame in self.frames.items():
            k=5 if dataset=='Helical' else 2
            original=runner.allocate_folds(frame,k).sort_values('recording').reset_index(drop=True)
            changed=frame.copy();changed[['delta_alpha','alpha0']]=1e9
            recomputed=runner.allocate_folds(changed,k).sort_values('recording').reset_index(drop=True)
            pd.testing.assert_frame_equal(original,recomputed)
            stored=frame[['recording','state','fold']].drop_duplicates().sort_values('recording').reset_index(drop=True)
            pd.testing.assert_frame_equal(original,stored)

    def test_transfer_missing_training_classes_not_dropped(self):
        split=pd.read_csv(OUT/'tables/operating_condition_splits.csv')
        pred=pd.read_csv(OUT/'tables/operating_condition_predictions.csv')
        self.assertTrue(pred.groupby(['dataset','channel','classifier','protocol','recording']).size().eq(1).all())
        for row in split.itertuples():
            frame=self.frames[row.dataset];frame=frame[frame.channel.eq(row.channel)]
            column='speed' if row.protocol=='leave_one_speed_out' else 'load'
            value=int(row.held_out_condition) if column=='speed' else row.held_out_condition
            train=frame[frame[column].ne(value)];test=frame[frame[column].eq(value)]
            missing=set(frame.state)-set(train.state)
            group=pred[pred.dataset.eq(row.dataset)&pred.channel.eq(row.channel)&pred.classifier.eq(row.classifier)&pred.protocol.eq(row.protocol)&pred.held_out_condition.astype(str).eq(str(value))]
            if missing:
                self.assertEqual(row.status,'NOT EVALUABLE');self.assertEqual(len(group),0)
            else:
                self.assertEqual(row.status,'EVALUABLE');self.assertEqual(set(group.recording),set(test.recording))

if __name__=='__main__':unittest.main(verbosity=2)
