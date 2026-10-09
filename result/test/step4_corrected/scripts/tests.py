"""Regression and behavioral checks for label-independent scaling and QC."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dontwrite_bytecode=True
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'code/core'))
import io
import json
import unittest
import numpy as np
from mfdfa import mfdfa,multifractal_spectrum
from scaling import fit_scaling,select_common,CANDIDATES,RULE
from vmd_sensitivity import aggregate,mode_metrics
from vmd import vmd as reference_vmd
from vmd_accel import vmd as accelerated_vmd,LIB
import warnings
from unittest.mock import patch
import vmd_sensitivity


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.q=np.arange(-5.,5.01,.5)
        self.s=np.unique(np.rint(np.geomspace(64,8192,40)).astype(int))

    def test_known_power_law_and_nonpositive_qc(self):
        fq=np.exp(self.q[:,None]/20)*self.s[None,:]**.7
        arrays,metrics=fit_scaling(self.s,fq,self.q,(64,512))
        np.testing.assert_allclose(arrays['hq'],.7,atol=1e-12)
        self.assertTrue(metrics['scaling_valid'])
        self.assertEqual(metrics['scale_points'],17)
        np.testing.assert_allclose(arrays['alpha'],.7,atol=1e-12)
        np.testing.assert_allclose(arrays['f_alpha'],1.,atol=1e-12)
        _,bad=fit_scaling(self.s,fq/self.s,self.q,(64,512))
        self.assertFalse(bad['scaling_valid'])
        self.assertIn('hq_near_zero_or_nonpositive',bad['invalid_reason'])
        self.assertEqual(bad['nonpositive_hq_count'],21)

    def test_matches_unchanged_core_fit(self):
        x=np.random.default_rng(119).normal(size=65536)
        s,fq,h,tau=mfdfa(x,self.q,scales=self.s,fit_range=(64,512))
        arrays,_=fit_scaling(s,fq,self.q,(64,512))
        np.testing.assert_allclose(arrays['hq'],h,atol=1e-12)
        np.testing.assert_allclose(arrays['tau'],tau,atol=1e-12)
        alpha,spectrum=multifractal_spectrum(self.q,tau)
        np.testing.assert_allclose(arrays['alpha'],alpha,atol=1e-12)
        np.testing.assert_allclose(arrays['f_alpha'],spectrum,atol=1e-12)

    def test_crossover_and_few_points_invalid(self):
        fq=np.minimum(self.s,400.)[None,:]**.7*np.ones((21,1))
        _,bad=fit_scaling(self.s,fq,self.q,(128,1024))
        self.assertFalse(bad['scaling_valid'])
        self.assertTrue('r2' in bad['invalid_reason'] or 'unstable_local_slopes' in bad['invalid_reason'])
        _,few=fit_scaling(self.s,self.s[None,:]**.7*np.ones((21,1)),self.q,(64,100))
        self.assertIn('insufficient_scale_points',few['invalid_reason'])

    def test_selection_does_not_use_labels_or_row_order(self):
        rows=[]
        for bounds in CANDIDATES:
            for i in range(16):
                _,metrics=fit_scaling(self.s,self.s[None,:]**.7*np.ones((21,1)),self.q,bounds)
                rows.append(dict(candidate_min=bounds[0],candidate_max=bounds[1],recording=str(i),label='Healthy',**metrics))
        chosen,_=select_common(rows)
        changed=[dict(r,label='different',class_separation=1e20) for r in rows[::-1]]
        again,_=select_common(changed)
        self.assertEqual(chosen,again)

    def test_no_common_valid_range_not_falsely_reported(self):
        rows=[]
        for low,high in CANDIDATES:
            _,metrics=fit_scaling(self.s,self.s[None,:]**.01*np.ones((21,1)),self.q,(low,high))
            rows.append(dict(candidate_min=low,candidate_max=high,**metrics))
        selected,_=select_common(rows)
        self.assertFalse(selected['common_valid'])


class ModeMetricTests(unittest.TestCase):
    def test_vmd_selection_ignores_labels_and_matches_frequency_not_index(self):
        results=[]
        for alpha in [500,1000]:
            summaries,centers=[],[]
            for channel in ['output_voltage','input_voltage']:
                for repeat in [1,2]:
                    identity=dict(configuration=f'K4_a{alpha}_t0',K=4,alpha=alpha,tau=0.,
                        recording=f'acquisition_{repeat}',acquisition_pair='acquisition',channel=channel,repeat=repeat)
                    summaries.append(dict(**identity,converged=alpha==500,iterations=100,
                        reconstruction_error=.05,constraint_error=.05,duplicate_fraction=0.,label='original'))
                    frequencies=[100.,200.,300.,400.] if repeat==1 else [400.,300.,200.,100.]
                    centers.extend(dict(**identity,mode_rank=i+1,center_hz=f,mode_energy_fraction=.25) for i,f in enumerate(frequencies))
            results.append(dict(summaries=summaries,centers=centers))
        captured={}
        def capture(path,rows): captured[path.name]=rows
        with patch.object(vmd_sensitivity,'table',side_effect=capture),patch.object(vmd_sensitivity,'dump'):
            chosen=aggregate(results)
            for item in results:
                item['summaries'].reverse(); item['centers'].reverse()
                for row in item['summaries']:
                    row.update(label='changed',delta_alpha=1000.,class_separation=1e9 if row['alpha']==1000 else -1e9)
            again=aggregate(results[::-1])
            self.assertEqual(chosen['selected_config'],again['selected_config'])
            self.assertEqual(chosen['selected_config']['alpha'],500)
            matches=captured['vmd_repeat_frequency_stability.csv']
            self.assertTrue(all(r['relative_center_difference']==0. for r in matches))
            self.assertTrue(any(r['repeat1_mode_rank']!=r['repeat2_mode_rank'] for r in matches))

    def test_energy_centers_bandwidth_correlation(self):
        fs=66666.7; n=4096; t=np.arange(n)/fs
        modes=np.array([np.cos(2*np.pi*(20*fs/n)*t),.5*np.cos(2*np.pi*(200*fs/n)*t)])
        x=modes.sum(axis=0)
        rows,quality=mode_metrics(x,modes,np.array([20*fs/n,200*fs/n]))
        np.testing.assert_allclose([r['observed_centroid_hz'] for r in rows],[20*fs/n,200*fs/n],atol=1e-6)
        np.testing.assert_allclose([r['mode_energy_fraction'] for r in rows],[.8,.2],atol=1e-12)
        self.assertLess(max(r['bandwidth_hz'] for r in rows),1e-6)
        self.assertEqual(quality['duplicate_fraction'],0)
        duplicate=np.array([modes[0],modes[0]])
        _,quality=mode_metrics(duplicate.sum(axis=0),duplicate,np.array([20*fs/n]*2))
        self.assertEqual(quality['duplicate_fraction'],1.)


@unittest.skipIf(LIB is None,'Native accelerator unavailable; NumPy fallback is used')
class AcceleratorTests(unittest.TestCase):
    def test_matches_reference_even_odd_dc_tau_and_random_init(self):
        for n,tau,dc,init in [(511,0.,False,1),(512,.5,False,1),(513,.5,True,1),
                             (512,0.,False,2),(512,.5,False,0)]:
            x=np.random.default_rng(65).normal(size=n)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                reference=reference_vmd(x,K=4,alpha=500,tau=tau,DC=dc,init=init,max_iter=100,return_info=True)
                actual=accelerated_vmd(x,K=4,alpha=500,tau=tau,DC=dc,init=init,max_iter=100,return_info=True)
            np.testing.assert_allclose(actual[0],reference[0],rtol=1e-9,atol=1e-10)
            np.testing.assert_allclose(actual[2],reference[2],rtol=1e-9,atol=1e-10)
            self.assertEqual(actual[3]['converged'],reference[3]['converged'])
            self.assertEqual(actual[3]['iterations'],reference[3]['iterations'])

    def test_zero_and_single_admm_step(self):
        reference=reference_vmd(np.zeros(101),K=2,return_info=True)
        actual=accelerated_vmd(np.zeros(101),K=2,return_info=True)
        np.testing.assert_array_equal(actual[0],reference[0])
        x=np.cos(2*np.pi*23*np.arange(1001)/1001)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            reference=reference_vmd(x,K=3,max_iter=1,return_info=True)
            actual=accelerated_vmd(x,K=3,max_iter=1,return_info=True)
        np.testing.assert_allclose(actual[0],reference[0],atol=1e-12)
        np.testing.assert_allclose(actual[2],reference[2],atol=1e-12)


if __name__=='__main__':
    log=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (BASE/'diagnostics/tests.log').write_text(log.getvalue(),encoding='utf-8')
    (BASE/'diagnostics/tests.json').write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2),encoding='utf-8')
    print(log.getvalue())
    sys.exit(0 if result.wasSuccessful() else 1)
