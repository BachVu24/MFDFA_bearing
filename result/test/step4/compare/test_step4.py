"""Checks for the practical long-record EMD variant; no writes outside compare."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dontwrite_bytecode=True
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[3]/'code/core'))
import io
import json
import unittest
import numpy as np
from emd_long_record import emd_long_record


class PracticalEMDTests(unittest.TestCase):
    def test_known_tones_reconstruction_and_peak(self):
        t=np.arange(8192)/4096
        x=np.cos(2*np.pi*32*t)+0.4*np.cos(2*np.pi*256*t)
        modes,residue,info=emd_long_record(x,max_imfs=2)
        self.assertEqual(modes.shape,(2,len(x)))
        np.testing.assert_allclose(modes.sum(axis=0)+residue,x,atol=1e-14)
        frequencies=np.fft.rfftfreq(len(x),1/4096)
        peaks=[frequencies[np.argmax(abs(np.fft.rfft(mode)))] for mode in modes]
        np.testing.assert_allclose(peaks,[256,32],atol=.5)
        self.assertTrue(info['all_practical_modes_converged'])

    def test_noisy_record_reports_approximation(self):
        x=np.random.default_rng(48).normal(size=65536)
        modes,residue,info=emd_long_record(x,max_imfs=3)
        self.assertEqual(len(modes),3)
        np.testing.assert_allclose(modes.sum(axis=0)+residue,x,atol=1e-14)
        for row in info['modes']:
            mismatch=abs(row['extrema']-row['zero_crossings'])
            self.assertEqual(row['exact_one_count_condition'],mismatch<=1)
            if row['converged_practical_rule']:
                self.assertLessEqual(mismatch,max(1,.005*row['extrema']))
                self.assertLessEqual(row['energy_sd'],.2)
                self.assertLessEqual(row['envelope_mean_ratio'],.05)

    def test_cap_is_visible(self):
        x=np.random.default_rng(52).normal(size=4096)
        modes,residue,info=emd_long_record(x,max_imfs=1,max_siftings=1)
        self.assertFalse(info['all_practical_modes_converged'])
        self.assertEqual(info['modes'][0]['status'],'sifting_cap_candidate')
        np.testing.assert_allclose(modes.sum(axis=0)+residue,x,atol=1e-14)

    def test_zero_and_monotone(self):
        for x in (np.zeros(128),np.arange(128.)):
            modes,residue,info=emd_long_record(x)
            self.assertEqual(modes.shape,(0,len(x)))
            np.testing.assert_array_equal(residue,x)


if __name__=='__main__':
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PracticalEMDTests))
    (OUT/'practical_emd_test_log.txt').write_text(log.getvalue(),encoding='utf-8')
    (OUT/'practical_emd_tests.json').write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),passed=result.wasSuccessful()),indent=2),encoding='utf-8')
    print(log.getvalue())
    sys.exit(0 if result.wasSuccessful() else 1)
