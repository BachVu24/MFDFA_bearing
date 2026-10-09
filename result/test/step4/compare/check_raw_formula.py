"""Independent spot check of real-data MF-DFA moments on selected scales."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dontwrite_bytecode=True
from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
checks=[]
for h,repeat,column in [(1,1,1),(5,2,1),(6,2,0)]:
    rid=f'helical_{h}_50hz_High_{repeat}'
    channel='output_voltage' if column==1 else 'input_voltage'
    x=np.loadtxt(ROOT/'DATA/labeled'/f'helical {h}'/f'helical {h}_50hz_High_{repeat}.txt')[:,column]
    y=np.cumsum(x-x.mean())
    with np.load(OUT.parent/'raw_mfdfa'/(rid+'_'+channel+'.npz')) as baseline:
        for target in (64,512,8192):
            si=int(np.argmin(abs(baseline['scales']-target)))
            s=int(baseline['scales'][si]); ns=len(x)//s
            variances=[]
            t=np.arange(s,dtype=float)
            for end in (False,True):
                for v in range(ns):
                    start=len(x)-(v+1)*s if end else v*s
                    segment=y[start:start+s]
                    trend=np.polyval(np.polyfit(t,segment,2),t)
                    variances.append(np.mean((segment-trend)**2))
            f2=np.asarray(variances)
            for q in (-5.,0.,5.):
                qi=int(np.flatnonzero(baseline['q']==q)[0])
                expected=np.exp(np.mean(np.log(f2))/2) if q==0 else np.mean(f2**(q/2))**(1/q)
                error=abs(expected-baseline['Fq'][qi,si])/expected
                assert error<1e-9,(rid,channel,s,q,error)
                checks.append(dict(recording=rid,channel=channel,scale=s,q=q,relative_error=float(error)))
(OUT/'raw_formula_validation.json').write_text(json.dumps(dict(checks_run=len(checks),all_passed=True,
    maximum_relative_error=max(c['relative_error'] for c in checks),checks=checks),indent=2),encoding='utf-8')
print('Independent raw formula checks:',len(checks),'passed; maximum relative error:',max(c['relative_error'] for c in checks))
