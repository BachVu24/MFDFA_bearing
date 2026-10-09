"""Behavioral regression checks: immutable configs, valid-only summaries,
matching independent of labels, factorial effects, bootstrap reproducibility."""
import pipeline as P
import analysis as A
import factorial_model as F
import unittest,itertools,io,json
from unittest.mock import patch
import numpy as np

class Step6Tests(unittest.TestCase):
    def test_portable_vmd_matches_reference(self):
        import accelerator
        accelerator.initialize(P.BASE/'_test_runtime')
        from vmd import vmd as reference
        x=np.random.default_rng(605).normal(size=2049)
        config=P.lock_config()['vmd']
        a=accelerator.vmd(x,**config,fs=66666.7,return_info=True)
        b=reference(x,**config,fs=66666.7,return_info=True)
        np.testing.assert_allclose(a[0],b[0],rtol=2e-9,atol=2e-10)
        np.testing.assert_allclose(a[2][-1],b[2][-1],rtol=2e-9,atol=2e-8)
        self.assertEqual(a[-1]['converged'],b[-1]['converged'])
        self.assertEqual(a[-1]['iterations'],b[-1]['iterations'])

    def test_parser_scope_and_independent_acquisitions(self):
        self.assertEqual(P.parse_name('helical 5_35hz_Low_2.txt'),('H5',35,'Low',2))
        for name in ['spur 1_35hz_High_1.txt','helical 1_25hz_High_1.txt','helical 1_35hz_High_3.txt']:
            self.assertIsNone(P.parse_name(name))
        keys={P.parse_name(f'helical {s[1:]}_{v}hz_{l}_{r}.txt') for s,v,l,r in itertools.product(P.STATES,P.SPEEDS,P.LOADS,[1,2])}
        self.assertEqual(len(keys),120)

    def test_locked_configuration_and_QC(self):
        config=P.lock_config()
        self.assertEqual(config['vmd'],dict(K=7,alpha=500,tau=0.,DC=False,init=1,tol=1e-7,max_iter=2000))
        self.assertEqual(config['qc_rule'],P.RULE)
        self.assertEqual(config['emd']['max_siftings'],200)
        self.assertEqual(sum(64<=s<=512 for s in config['scales']),17)

    def test_band_selection_before_QC_no_invalid_rescue(self):
        source=dict(recording='helical_1_30hz_High_1',state='H1',speed=30,load='High',repeat=1)
        rows=[]
        for component,energy,valid in [('mode1',10.,False),('mode2',1.,True)]:
            rows.append(dict(source,channel='output_voltage',pipeline='vmd',component=component,
                center_hz=500.,mode_energy=energy,analysis_valid=valid,decomposition_valid=True,invalid_reason='QC fail' if not valid else ''))
        captured={}
        with patch.object(P,'table',side_effect=lambda path,data:captured.update({path.name:data})):
            reps,coverage=A.representatives(rows,[source])
        selected=next(r for r in reps if r['pipeline']=='vmd' and r['channel']=='output_voltage' and r['category']=='band_0_1000_Hz')
        self.assertEqual(selected['component'],'mode1');self.assertFalse(selected['analysis_valid'])
        self.assertTrue(any(r['status']=='NOT EVALUABLE' for r in coverage))

    def test_invalid_repeat_excluded_and_not_pseudoreplicated(self):
        reps=[]
        for repeat,valid in [(1,True),(2,False)]:
            reps.append(dict(recording=f'acq_{repeat}',pipeline='raw',channel='output_voltage',category='raw',
                state='H1',speed=30,load='High',repeat=repeat,analysis_valid=valid,delta_alpha=.5,alpha0=1.,delta_h=.2))
        with patch.object(P,'table'):
            cells=A.build_cells(reps)
        cell=next(c for c in cells if c['state']=='H1' and c['speed']==30 and c['load']=='High')
        self.assertFalse(cell['evaluable']);self.assertNotIn('delta_alpha_mean',cell)
        self.assertEqual(cell['valid_count'],1)

    def test_matching_frequency_not_rank_or_label(self):
        rows=[]
        for repeat,frequencies in [(1,[100.,400.]),(2,[400.,100.])]:
            for rank,freq in enumerate(frequencies,1):
                rows.append(dict(recording=f'acq_{repeat}',pipeline='vmd',channel='output_voltage',component=f'mode{rank}',
                    repeat=repeat,center_hz=freq,analysis_valid=True,label='arbitrary'))
        captured={}
        with patch.object(P,'table',side_effect=lambda path,data:captured.update({path.name:data})):
            A.mode_matching_audit(rows)
            matches=captured['repeat_frequency_matching.csv']
        self.assertTrue(all(r['relative_frequency_drift']==0 for r in matches))
        self.assertTrue(any(r['repeat1_component']!=r['repeat2_component'] for r in matches))

    def test_factorial_identifies_known_effect_and_full_design(self):
        rows=[dict(state=s,speed=v,load=l,repeat=r,recording=f'{s}_{v}_{l}_{r}') for s,v,l,r in itertools.product(P.STATES,P.SPEEDS,P.LOADS,[1,2])]
        X,indices=F.design(rows);full,_=F.design(rows,True)
        self.assertEqual(X.shape,(120,11));self.assertEqual(full.shape,(120,60));self.assertEqual(np.linalg.matrix_rank(full),60)
        y=5*X[:,indices['fault'][0]]+np.random.default_rng(11).normal(scale=.2,size=120)
        fault=F.term_test(X,y,indices['fault'],B=199);speed=F.term_test(X,y,indices['speed'],B=199)
        self.assertGreater(fault['partial_eta_squared'],.95)
        self.assertGreater(fault['incremental_r_squared'],speed['incremental_r_squared'])
        self.assertLessEqual(fault['wild_bootstrap_p'],.01)
        self.assertEqual(fault,F.term_test(X,y,indices['fault'],B=199))
        inv,b,e,h=F.ols(X,y);ss=(e@e)
        reduced=np.delete(X,indices['fault'],axis=1);e0=y-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0]
        self.assertAlmostEqual(fault['term_ss'],e0@e0-ss,places=10)

    def test_sparse_factorial_not_evaluable(self):
        rows=[dict(state='H1',speed=30,load='High',recording=str(i),delta_alpha=.2+i*.01) for i in range(10)]
        results,diagnostics=F.factorial(rows,'delta_alpha',B=19)
        self.assertTrue(all(r['status']=='NOT EVALUABLE' for r in results));self.assertFalse(diagnostics)

    def test_complete_factorial_descriptive_summaries_and_absent_bands(self):
        sources=[];rows=[]
        for state,speed,load,repeat in itertools.product(P.STATES,P.SPEEDS,P.LOADS,[1,2]):
            source=dict(recording=f'helical_{state[1:]}_{speed}hz_{load}_{repeat}',state=state,speed=speed,load=load,repeat=repeat)
            sources.append(source)
            for _,channel,_ in P.CHANNELS:
                for method,component in [('raw','raw'),('vmd','oscillatory_sum'),('emd','oscillatory_sum')]:
                    value=.2+.1*P.STATES.index(state)+.001*speed+.02*(load=='Low')+.005*repeat
                    rows.append(dict(source,channel=channel,pipeline=method,component=component,analysis_valid=True,
                        decomposition_valid=True,invalid_reason='',center_hz='',delta_alpha=value,alpha0=1+value,delta_h=value/2))
        with patch.object(P,'table'):
            reps,coverage=A.representatives(rows,sources);cells=A.build_cells(reps);pairs=A.comparisons(cells)
            speeds,loads=A.condition_variation(cells);effects=A.fault_operating_effect(cells,pairs)
            paired,summary=A.paired_raw_vmd(reps)
            grid=A.matched_operating_grid(cells)
        main=[r for r in cells if r['pipeline']=='raw' and r['channel']=='output_voltage']
        self.assertEqual(sum(r['evaluable'] for r in main),60)
        self.assertTrue(all(not r['evaluable'] for r in cells if r['category'].startswith('band_')))
        self.assertEqual(len([r for r in pairs if r['pipeline']=='raw' and r['channel']=='output_voltage' and r['evaluable']]),450)
        self.assertTrue(all(r['evaluable'] for r in speeds if r['pipeline']=='raw'))
        self.assertTrue(all(r['evaluable'] for r in loads if r['pipeline']=='raw'))
        self.assertEqual(len(paired),720)
        self.assertTrue(all(abs(r['pearson_r']-1)<1e-12 for r in summary))
        r=next(r for r in grid if (r['pipeline'],r['channel'],r['feature'])==('raw','output_voltage','delta_alpha'))
        self.assertEqual(r['common_operating_cells'],10)
        operating=np.array([.001*s+.02*(l=='Low') for s,l in itertools.product(P.SPEEDS,P.LOADS)])
        self.assertAlmostEqual(r['fault_to_operating_ratio'],.2/operating.std(ddof=1),places=8)

if __name__=='__main__':
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__('sys').modules[__name__]))
    output=P.Path(P.os.environ.get('PHM_TEST_OUTPUT_DIR',str(P.BASE/'diagnostics')))
    output.mkdir(parents=True,exist_ok=True)
    (output/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    P.dump(output/'tests.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    print(log.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
