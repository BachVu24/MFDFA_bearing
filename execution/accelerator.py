"""OS-aware loader for the immutable Step 4 accelerator function.

Extract only its function AST: do not execute the historical Windows DLL loader.
The recurrence and wrapper stay single-sourced and retain their locked hashes.
"""
from pathlib import Path
import ast, ctypes, os, platform, shutil, subprocess, warnings
import numpy as np
from execution.provenance import ROOT, atomic_json, sha
import sys
sys.path.insert(0,str(ROOT/'code/core'))
from vmd import vmd as reference_vmd
SOURCE=ROOT/'result/test/step4_corrected/scripts/vmd_accel.py'
CPP=SOURCE.with_name('vmd_kernel.cpp')
LIB=None
STATUS={'backend':'numpy_reference','reason':'not initialized'}
function=next(n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='vmd')
namespace=dict(np=np,warnings=warnings,reference_vmd=reference_vmd,LIB=None)
exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),namespace)
_native=namespace['vmd']
_handle=None
def initialize(directory, compile_linux=True):
    global LIB, STATUS, _handle
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    namespace['LIB']=None;LIB=None
    status=dict(backend='numpy_reference',platform=platform.platform(),source_sha256=sha(SOURCE),cpp_sha256=sha(CPP),parity_passed=False)
    try:
        if platform.system()=='Windows':
            library=SOURCE.parents[1]/'_runtime/vmd_kernel.dll'
            if hasattr(os,'add_dll_directory'): _handle=os.add_dll_directory(str(library.parent))
        elif platform.system()=='Linux':
            library=directory/'vmd_kernel.so'
            compiler=shutil.which('g++')
            if not compile_linux or compiler is None: raise RuntimeError('Linux compilation disabled or g++ unavailable')
            command=[compiler,'-O3','-std=c++17','-shared','-fPIC','-D__declspec(x)=',str(CPP),'-o',str(library)]
            proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
            status.update(compile_command=command,compile_returncode=proc.returncode,compile_output=proc.stdout+proc.stderr)
            if proc.returncode: raise RuntimeError('Linux compilation failed')
        else: raise RuntimeError('Unsupported native platform')
        lib=ctypes.CDLL(str(library))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.vmd_admm.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ptr,ptr,ptr,ptr]
        lib.vmd_admm.restype=ctypes.c_int;namespace['LIB']=lib
        cases=[];rng=np.random.default_rng(6509)
        for length,tau,dc,init in [(512,0,False,1),(513,.5,False,2),(512,0,True,0),(512,.5,True,1),(1024,0,False,1)]:
            x=rng.normal(size=length);kwargs=dict(K=7,alpha=500,tau=tau,DC=dc,init=init,tol=1e-7,max_iter=2000 if length==1024 else 100,return_info=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');a=reference_vmd(x,**kwargs);b=_native(x,**kwargs)
            errors=[float(np.linalg.norm(v-w)/max(np.linalg.norm(v),1e-15)) for v,w in zip(a[:3],b[:3])]
            passed=max(errors)<1e-9 and a[-1]['iterations']==b[-1]['iterations'] and a[-1]['converged']==b[-1]['converged']
            cases.append(dict(length=length,tau=tau,DC=dc,init=init,relative_errors=errors,passed=passed))
        status['parity_cases']=cases
        if not all(x['passed'] for x in cases): raise RuntimeError('Numerical parity failed')
        LIB=lib;status.update(backend='cpp_fused_admm',parity_passed=True,library=str(library),library_sha256=sha(library))
    except Exception as error:
        namespace['LIB']=None;status['reason']=str(error)
    STATUS=status;atomic_json(directory/'accelerator_status.json',status);return status

def vmd(*args,**kwargs):
    return _native(*args,**kwargs)
