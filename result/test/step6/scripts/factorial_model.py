"""Recording-level exploratory factorial regression, HC3 and wild bootstrap.

No pipeline settings are selected here. Sum-to-zero orthogonal contrasts;
main effects in additive model, interaction terms in complete full model.
"""
import numpy as np
from scipy import stats,linalg

LEVELS={'fault':['H1','H2','H3','H4','H5','H6'],'speed':[30,35,40,45,50],'load':['High','Low']}

def design(rows,full=False):
    blocks={}
    for term,levels in LEVELS.items():
        key='state' if term=='fault' else term
        contrasts=linalg.helmert(len(levels),full=False).T
        blocks[term]=contrasts[[levels.index(r[key]) for r in rows]]
    if full:
        for terms in [('fault','speed'),('fault','load'),('speed','load'),('fault','speed','load')]:
            value=blocks[terms[0]]
            for term in terms[1:]:value=(value[:,:,None]*blocks[term][:,None,:]).reshape(len(rows),-1)
            blocks[':'.join(terms)]=value
    X=np.ones((len(rows),1));indices={}
    for term,block in blocks.items():
        indices[term]=np.arange(X.shape[1],X.shape[1]+block.shape[1]);X=np.c_[X,block]
    return X,indices

def ols(X,y):
    inverse=np.linalg.pinv(X);beta=inverse@y;residual=y-X@beta
    leverage=np.einsum('ij,ji->i',X,inverse)
    return inverse,beta,residual,leverage

def wald(inverse,beta,residual,leverage,index):
    selected=inverse[index]/np.maximum(1-leverage,1e-10)
    covariance=(selected*residual**2)@selected.T
    b=beta[index]
    return float(b@np.linalg.pinv(covariance)@b)

def term_test(X,y,index,B=1999,seed=6509):
    inverse,beta,e,h=ols(X,y);n,p=X.shape
    reduced=np.delete(X,index,axis=1);inv0,b0,e0,h0=ols(reduced,y)
    sse=float(e@e);sse0=float(e0@e0);increment=max(sse0-sse,0.)
    df=len(index);F=(increment/df)/(sse/(n-p)) if sse else np.inf
    W=wald(inverse,beta,e,h,index)
    rng=np.random.default_rng(seed)
    signs=rng.choice([-1.,1.],size=(n,B))
    Y=reduced@b0[:,None]+(e0/np.sqrt(np.maximum(1-h0,1e-10)))[:,None]*signs
    betas=inverse@Y;errors=Y-X@betas
    A=inverse[index]/np.maximum(1-h,1e-10)
    covariance=np.einsum('in,jn,nb->bij',A,A,errors**2,optimize=True)
    b=betas[index].T
    Ws=np.einsum('bi,bij,bj->b',b,np.linalg.pinv(covariance),b)
    return dict(term_ss=increment,partial_eta_squared=increment/(increment+sse) if increment+sse else 0.,
        incremental_r_squared=increment/np.sum((y-y.mean())**2) if np.ptp(y) else 0.,
        classical_F=float(F),classical_p=float(stats.f.sf(F,df,n-p)),HC3_Wald=float(W),
        HC3_asymptotic_p=float(stats.chi2.sf(W,df)),wild_bootstrap_p=float((1+np.sum(Ws>=W))/(B+1)),
        bootstrap_draws=B,term_df=df,residual_df=n-p)

def factorial(rows,feature,B=1999):
    y=np.array([r[feature] for r in rows],float);output=[];diagnostics=[]
    roster=';'.join(r['recording'] for r in rows)
    for full in [False,True]:
        model='fault*speed*load' if full else 'fault+speed+load'
        X,indices=design(rows,full);rank=np.linalg.matrix_rank(X)
        reason=''
        if len(rows)<30 or rank<X.shape[1] or len(rows)-rank<20:reason='insufficient coverage/rank/residual_df'
        if full and len({(r['state'],r['speed'],r['load']) for r in rows})<60:reason='missing factorial cells'
        if reason:
            output.append(dict(model=model,term='all',status='NOT EVALUABLE',reason=reason,n=len(rows),source_recordings=roster));continue
        inverse,beta,e,h=ols(X,y)
        shap=stats.shapiro(e)
        # Koenker-style LM n*R2 of squared residuals on additive design.
        Xmain,_=design(rows);squared=e**2;pred=Xmain@np.linalg.lstsq(Xmain,squared,rcond=None)[0]
        total=np.sum((squared-squared.mean())**2)
        lm=len(rows)*(1-np.sum((squared-pred)**2)/total) if total else 0.
        bp=float(stats.chi2.sf(max(lm,0),Xmain.shape[1]-1))
        diagnostics.append(dict(model=model,n=len(rows),rank=rank,residual_df=len(rows)-rank,
            shapiro_p=float(shap.pvalue),heteroskedasticity_LM_p=bp,
            max_leverage=float(h.max()),residual_rms=float(np.sqrt(np.mean(e**2))),
            assumptions_flag=bool(shap.pvalue<.05 or bp<.05),source_recordings=roster))
        for term,index in indices.items():
            result=term_test(X,y,index,B=B)
            output.append(dict(model=model,term=term,status='exploratory',n=len(rows),**result,source_recordings=roster,
                interpretation='HC3/wild bootstrap primary; classical p reference only. Conditional fixed design, independence not empirically established.'))
    return output,diagnostics

def bh_adjust(rows,field='wild_bootstrap_p'):
    valid=[r for r in rows if field in r];ordered=sorted(valid,key=lambda r:r[field]);m=len(ordered);ceiling=1.
    for i in range(m-1,-1,-1):
        ceiling=min(ceiling,ordered[i][field]*m/(i+1));ordered[i][field+'_BH']=ceiling
