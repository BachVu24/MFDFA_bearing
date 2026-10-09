"""Scientific figures, auditable report and final cleanup for Step 5."""
import pipeline as P
import numpy as np
import json,itertools,hashlib,re,shutil
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from analysis import FEATURES,METHODS,ANALYSIS_RULE,CONFIG

COLORS={'H1':'#2274a5','H2':'#e18d22','H3':'#8856a7','H4':'#d95f02','H5':'#bf3d56','H6':'#39935c'}
def save(fig,name):
    fig.tight_layout();fig.savefig(P.BASE/'figures'/name,dpi=150);plt.close(fig)
def main_category(method):return 'raw' if method=='raw' else 'oscillatory_sum'

def figures(features,reps,cells,coverage,paired,models):
    primary='output_voltage'
    for method in METHODS:
        category=main_category(method)
        subset=[c for c in cells if c['pipeline']==method and c['channel']==primary and c['category']==category]
        complete=sum(c['evaluable'] for c in subset)
        if method=='emd' and complete<45:continue
        fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
        for i,f in enumerate(FEATURES):
            for j,load in enumerate(P.LOADS):
                ax=axes[i,j]
                for state in P.STATES:
                    rows=sorted([c for c in subset if c['state']==state and c['load']==load],key=lambda c:c['speed'])
                    means=[c[f+'_mean'] if c['evaluable'] else np.nan for c in rows]
                    deviations=[c[f+'_std'] if c['evaluable'] else np.nan for c in rows]
                    ax.errorbar(P.SPEEDS,means,yerr=deviations,color=COLORS[state],marker='o',capsize=3,label=state)
                ax.set(title=load+' / '+f,xlabel='Rotational speed (Hz)',ylabel=f);ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle(method.upper()+'/'+category+' / column 2; mean ± acquisition SD, gaps = NOT EVALUABLE')
        save(fig,'feature_vs_speed_'+method+'.png')
    # Per-feature heatmaps for main representations, both channels explicitly.
    for _,ch,_ in P.CHANNELS:
        fig,axes=plt.subplots(3,3,figsize=(16,10))
        color_limits={}
        for f in FEATURES:
            values=[c[f+'_mean'] for c in cells if c['channel']==ch and c['category']==main_category(c['pipeline']) and c['evaluable']]
            color_limits[f]=(min(values),max(values)) if values else (0,1)
        for i,method in enumerate(METHODS):
            for j,f in enumerate(FEATURES):
                ax=axes[i,j];matrix=np.full((6,10),np.nan)
                for si,state in enumerate(P.STATES):
                    for ci,(speed,load) in enumerate(itertools.product(P.SPEEDS,P.LOADS)):
                        c=next(c for c in cells if (c['pipeline'],c['channel'],c['category'],c['state'],c['speed'],c['load'])==(method,ch,main_category(method),state,speed,load))
                        if c['evaluable']:matrix[si,ci]=c[f+'_mean']
                image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap='viridis',vmin=color_limits[f][0],vmax=color_limits[f][1])
                ax.set_xticks(range(10),[f'{s}/{l[0]}' for s,l in itertools.product(P.SPEEDS,P.LOADS)],rotation=45)
                ax.set_yticks(range(6),P.STATES);ax.set_title(method.upper()+' '+f);fig.colorbar(image,ax=ax,shrink=.75)
                for si,ci in zip(*np.where(np.isnan(matrix))):ax.text(ci,si,'NE',ha='center',va='center',fontsize=7,color='red')
        fig.suptitle(ch+' / cell means, NE = NOT EVALUABLE')
        save(fig,'feature_heatmaps_'+ch+'.png')
    # Recording-level distributions across speeds (not windows).
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharey='col')
    for i,method in enumerate(METHODS):
        for j,f in enumerate(FEATURES):
            ax=axes[i,j];labels=[];data=[]
            for state,load in itertools.product(P.STATES,P.LOADS):
                values=[r[f] for r in reps if (r['pipeline'],r['channel'],r['category'],r['state'],r['load'])==(method,primary,main_category(method),state,load) and r['analysis_valid']]
                labels.append(state+'/'+load[0]);data.append(values or [np.nan])
            ax.boxplot(data,tick_labels=labels,showfliers=False)
            for k,values in enumerate(data,1):ax.scatter(np.full(len(values),k),values,s=9,alpha=.6)
            ax.tick_params(axis='x',rotation=45);ax.set(title=method.upper()+' '+f,ylabel=f);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Column 2: recording distributions across all speeds; QC-valid only')
    save(fig,'feature_distributions.png')
    for _,ch,_ in P.CHANNELS:
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,f in zip(axes,FEATURES):
            for state in P.STATES:
                for load,marker in [('High','o'),('Low','^')]:
                    subset=[r for r in paired if r['channel']==ch and r['feature']==f and r['state']==state and r['load']==load and r['evaluable']]
                    ax.scatter([r['raw_feature'] for r in subset],[r['vmd_feature'] for r in subset],s=25,color=COLORS[state],marker=marker,label=state+'/'+load[0])
            limits=ax.get_xlim()+ax.get_ylim();lo=min(limits);hi=max(limits)
            ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set(title=f,xlabel='Raw feature',ylabel='VMD oscillatory_sum feature');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle('Paired full-recording features / '+ch)
        save(fig,'paired_raw_vmd_'+ch+'.png')
    # Complete-cell coverage includes every band and all 40 operating/fault cells.
    for _,ch,_ in P.CHANNELS:
        identities=sorted({(r['pipeline'],r['category']) for r in coverage if r['channel']==ch})
        matrix=np.zeros((len(identities),60))
        for ri,(method,category) in enumerate(identities):
            for ci,(state,speed,load) in enumerate(itertools.product(P.STATES,P.SPEEDS,P.LOADS)):
                r=next(r for r in coverage if (r['pipeline'],r['channel'],r['category'],r['state'],r['speed'],r['load'])==(method,ch,category,state,speed,load))
                matrix[ri,ci]=r['valid_fraction']
        fig,ax=plt.subplots(figsize=(18,8));im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set_yticks(range(len(identities)),[m+'/'+c for m,c in identities]);ax.set_xticks(range(60),[st+':'+str(s)+'/'+l[0] for st,s,l in itertools.product(P.STATES,P.SPEEDS,P.LOADS)],rotation=90,fontsize=6)
        ax.set_title('QC-valid acquisition fraction / '+ch+'; matching eligibility in qc_coverage.csv');fig.colorbar(im,ax=ax,label='valid / 2 expected')
        save(fig,'qc_coverage_'+ch+'.png')
    # Assumption diagnostics for main additive representations, primary channel.
    from factorial_model import design,ols
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,method in enumerate(METHODS):
        rows=[r for r in reps if (r['pipeline'],r['channel'],r['category'])==(method,primary,main_category(method)) and r['analysis_valid']]
        if len(rows)<30:continue
        X,_=design(rows)
        y=np.array([r['alpha0'] for r in rows]);inv,b,e,h=ols(X,y)
        axes[0,j].scatter(X@b,e,s=15);axes[0,j].axhline(0,color='k',lw=.7);axes[0,j].set(title=method.upper()+' alpha0 residuals',xlabel='Fitted',ylabel='Residual')
        stats.probplot(e,plot=axes[1,j]);axes[1,j].set_title(method.upper()+' alpha0 Q-Q')
    save(fig,'factorial_residual_diagnostics.png')

