"""Optional compiled fused-loop accelerator for the unchanged VMD recurrence.

Portable fallback is the original NumPy implementation. Build with:
g++ -O3 -std=c++17 -shared -static-libgcc -static-libstdc++ vmd_kernel.cpp
    -o ../_runtime/vmd_kernel.dll
No fast-math. Floating-point reduction order differs; parity is tested.
"""
from pathlib import Path
import ctypes
import os
import warnings
import numpy as np
from vmd import vmd as reference_vmd
BASE=Path(__file__).resolve().parents[1]
DLL=BASE/'_runtime/vmd_kernel.dll'
LIB=None
DLL_DIRECTORY_HANDLE=None
if DLL.exists():
    try:
        if hasattr(os,'add_dll_directory'): DLL_DIRECTORY_HANDLE=os.add_dll_directory(str(DLL.parent))
        LIB=ctypes.CDLL(str(DLL))
        PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        LIB.vmd_admm.argtypes=[PTR,PTR,ctypes.c_int,ctypes.c_int,PTR,
            ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,PTR,PTR,PTR,PTR]
        LIB.vmd_admm.restype=ctypes.c_int
    except OSError:
        LIB=None


def vmd(x,alpha=2000.,tau=.5,K=3,DC=False,init=1,tol=1e-7,*,max_iter=2000,
        fs=1.,seed=0,return_info=False):
    if LIB is None:
        result=reference_vmd(x,alpha,tau,K,DC,init,tol,max_iter=max_iter,fs=fs,seed=seed,return_info=return_info)
        if return_info: result[-1]['numeric_backend']='numpy_reference'
        return result
    raw=np.asarray(x)
    # Delegate uncommon invalid-input checks to the original implementation.
    if (np.iscomplexobj(raw) or raw.ndim!=1 or len(raw)<3 or not np.isfinite(raw).all()
            or not isinstance(K,(int,np.integer)) or K<1 or init not in (0,1,2)
            or not isinstance(max_iter,(int,np.integer)) or max_iter<1
            or not np.isfinite(tau) or tau<0 or not np.isfinite(tol) or tol<=0
            or not np.isfinite(fs) or fs<=0):
        return reference_vmd(x,alpha,tau,K,DC,init,tol,max_iter=max_iter,fs=fs,seed=seed,return_info=return_info)
    signal=np.asarray(raw,dtype=np.float64)
    penalties=np.asarray(alpha,dtype=np.float64)
    if penalties.ndim==0: penalties=np.full(K,float(penalties))
    if penalties.shape!=(K,) or not np.isfinite(penalties).all() or np.any(penalties<=0):
        return reference_vmd(x,alpha,tau,K,DC,init,tol,max_iter=max_iter,fs=fs,seed=seed,return_info=return_info)
    penalties=np.ascontiguousarray(penalties)
    amplitude=float(np.max(np.abs(signal))); normalized=signal/(amplitude if amplitude else 1.)
    half=len(signal)//2
    extended=np.r_[normalized[:half][::-1],normalized,normalized[-half:][::-1]]
    length=len(extended)
    frequencies=np.fft.rfftfreq(length)
    target=np.fft.rfft(extended)
    modes_hat=np.zeros((K,len(target)),dtype=np.complex128)
    centers=np.zeros(K,dtype=np.float64)
    if init==1: centers=.5*np.arange(K,dtype=np.float64)/K
    elif init==2:
        centers=np.sort(np.exp(np.random.default_rng(seed).uniform(np.log(1./length),np.log(.5),K)))
    if DC: centers[0]=0
    history=np.zeros((max_iter+1,K),dtype=np.float64)
    metrics=np.zeros(3,dtype=np.float64)
    iterations=LIB.vmd_admm(target.view(np.float64),frequencies,len(target),K,penalties,
        tau,tol,max_iter,int(bool(DC)),modes_hat.view(np.float64),centers,history,metrics)
    converged=bool(metrics[2])
    if not converged: warnings.warn('VMD reached max_iter before convergence',RuntimeWarning,stacklevel=2)
    modes=np.fft.irfft(modes_hat,n=length,axis=1)[:,half:half+len(signal)]*amplitude
    spectra=np.fft.fftshift(np.fft.fft(modes,axis=1),axes=1).T
    norm=np.linalg.norm(normalized)
    reconstruction=float(np.linalg.norm(normalized-modes.sum(axis=0)/(amplitude if amplitude else 1.))/norm) if norm else 0.
    info=dict(converged=converged,iterations=iterations,relative_change=float(metrics[0]),
        constraint_error=float(metrics[1]),reconstruction_error=reconstruction,
        frequency_unit='cycles per time unit',tau=float(tau),numeric_backend='cpp_fused_admm')
    result=(modes,spectra,history[:iterations+1]*fs)
    return (*result,info) if return_info else result
