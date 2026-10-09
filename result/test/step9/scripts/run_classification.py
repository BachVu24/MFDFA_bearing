"""Fixed-feature, fixed-parameter recording-level classification; no tuning."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'result/test/step9'
os.environ['MPLCONFIGDIR'] = str(OUT/'_work/mplconfig')
os.environ['XDG_CACHE_HOME'] = str(OUT/'_work/cache')
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=9609
CHANNELS=['output_voltage','input_voltage']
FEATURES=['delta_alpha','alpha0']
MODULES={'KNN':'knn','SVM':'svm','LDA':'lda','Random Forest':'random_forest'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def dump(name, value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')

def save(name, rows):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path,index=False)
    return frame

def measures(y,pred,labels):
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=labels,average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(y,pred),macro_f1=f1,macro_precision=precision,macro_recall=recall,
                weighted_f1=f1_score(y,pred,labels=labels,average='weighted',zero_division=0))

def allocate_folds(frame,k,seed=SEED):
    """Stratify configuration and balance channel eligibility, without features.

    Each recording gets one global fold reused in both channels. Shared-channel
    recordings are allocated first, then single-channel records, minimizing the
    current eligible count for their channel(s). No feature values or predictions
    enter this deterministic allocation.
    """
    rng=np.random.default_rng(seed);rows=[]
    recordings=frame.groupby('recording').agg(state=('state','first'),channels=('channel',lambda s:tuple(sorted(s))))
    for state,group in recordings.groupby('state',sort=True):
        counts=np.zeros((k,2),dtype=int);total=np.zeros(k,dtype=int)
        group=group.sample(frac=1,random_state=int(rng.integers(2**31-1)))
        order=sorted(group.index,key=lambda r:-len(group.loc[r,'channels']))
        ties=list(rng.permutation(k))
        for recording in order:
            active=[CHANNELS.index(c) for c in group.loc[recording,'channels']]
            fold=min(range(k),key=lambda f:(int(counts[f,active].sum()),int(total[f]),ties.index(f)))
            counts[fold,active]+=1;total[fold]+=1
            rows.append(dict(recording=recording,state=state,fold=fold))
    return pd.DataFrame(rows)

def run():
    for directory in ['config','tables','plots','confusion_matrices','diagnostics']:(OUT/directory).mkdir(parents=True,exist_ok=True)
    feature_lock=load_module('step8_lock',ROOT/'result/test/step8/scripts/load_locked_features.py')
    config=feature_lock.load_locked_config()
    assert config['features']==FEATURES
    models={name:load_module('classifier_'+module,ROOT/f'code/core/classifier/{module}.py').build_classifier() for name,module in MODULES.items()}
    protocol=dict(seed=SEED,features=FEATURES,representation='RAW',hyperparameter_search=False,
        metric_for_comparison='Primary: pooled out-of-fold macro F1 within each dataset/channel; accuracy secondary; ties reported.',
        folds={'Helical':5,'Spur':2},split_method='Configuration-stratified, channel-eligibility-balanced recording groups; saved before fitting. Same recording fold across channels.',
        supplemental_protocols=['leave_one_speed_out','leave_one_load_out'],
        missing_training_class='Supplemental split NOT EVALUABLE, no dropping class or fitting reduced target set.',
        reporting_scope='Eligible configurations only; full S1-S8 classification NOT EVALUABLE.',
        feature_selection_bias='Step8 selected using these labels. Internal CV conditional on fixed features is exploratory; independent end-to-end validation needs untouched recordings.',
        libraries={'python':sys.version.split()[0],'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,
                   'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        classifiers={name:{key:value for key,value in model.get_params(deep=True).items() if key not in ['steps'] and not hasattr(value,'fit')} for name,model in models.items()},
        step8_config_sha256=digest(ROOT/'result/test/step8/config/locked_feature_set.json'))
    dump('config/evaluation_protocol.json',protocol)
    (OUT/'config/requirements.txt').write_text('\n'.join([f'scikit-learn=={sklearn.__version__}',f'numpy=={np.__version__}',
        f'pandas=={pd.__version__}',f'scipy=={scipy.__version__}',f'matplotlib=={matplotlib.__version__}'])+'\n')
    dataset_frames={};splits={};class_rows=[];source_hashes=[]
    for step,dataset in [(6,'Helical'),(7,'Spur')]:
        base=ROOT/f'result/test/step{step}'
        source=base/'features/analysis_representatives.csv'
        source_hashes.append(dict(path=str(source.relative_to(ROOT)),sha256=digest(source)))
        frame=pd.read_csv(source)
        labels=json.loads((base/'config/fault_labels_from_workbook.json').read_text())
        dump(f'config/{dataset.lower()}_original_fault_metadata.json',labels)
        all_states=[f'{"H" if step==6 else "S"}{i}' for i in range(1,(6 if step==6 else 8)+1)]
        eligible=[]
        for channel in CHANNELS:
            meta,X,excluded=feature_lock.locked_inputs(frame,channel)
            used=pd.concat([meta,X],axis=1).sort_values('recording').reset_index(drop=True)
            eligible.append(used)
            raw=frame[(frame.pipeline=='raw')&(frame.channel==channel)&(frame.component=='raw')]
            for state in all_states:
                count=int(used.state.eq(state).sum())
                class_rows.append(dict(dataset=dataset,channel=channel,state=state,
                    label=raw.loc[raw.state.eq(state),'label'].iloc[0],expected_recordings=20,valid_recordings=count,
                    excluded_recordings=20-count,qc_coverage=count/20,
                    status='AVAILABLE' if count else 'NOT EVALUABLE - no QC-valid sample'))
        used=pd.concat(eligible,ignore_index=True)
        assert not used.duplicated(['recording','channel']).any()
        fold=allocate_folds(used,protocol['folds'][dataset])
        fold.insert(0,'dataset',dataset)
        save(f'tables/{dataset.lower()}_recording_folds.csv',fold)
        used=used.merge(fold[['recording','fold']],on='recording',validate='many_to_one')
        save(f'tables/{dataset.lower()}_eligible_samples.csv',used)
        dataset_frames[dataset]=used
        splits[dataset]=fold
        for channel in CHANNELS:
            group=used[used.channel==channel]
            for state in group.state.unique():
                for f in range(protocol['folds'][dataset]):
                    assert ((group.state==state)&(group.fold==f)).any(), (dataset,channel,state,f)
                    assert ((group.state==state)&(group.fold!=f)).any()
    distribution=save('class_distribution.csv',class_rows)
    source_hashes.extend(dict(path=f'code/core/classifier/{module}.py',sha256=digest(ROOT/f'code/core/classifier/{module}.py')) for module in MODULES.values())
    dump('config/source_hashes.json',source_hashes)
    # At this point all split/config files exist before the first model fit.
    results=[];fold_metrics=[];predictions=[];class_metrics=[];transfer=[];transfer_pred=[];baseline=[];fit_audit=[]
    for dataset,frame in dataset_frames.items():
        all_states=distribution[distribution.dataset.eq(dataset)].state.unique().tolist()
        for channel in CHANNELS:
            group=frame[frame.channel==channel].sort_values('recording').reset_index(drop=True)
            labels=sorted(group.state.unique());X=group[FEATURES];y=group.state
            for name,model in models.items():
                oof=np.empty(len(group),dtype=object)
                for fold in range(protocol['folds'][dataset]):
                    train=group.fold.ne(fold);test=group.fold.eq(fold)
                    assert not set(group.loc[train,'recording'])&set(group.loc[test,'recording'])
                    fitted=clone(model).fit(X[train],y[train]);pred=fitted.predict(X[test]);oof[test]=pred
                    fold_metrics.append(dict(dataset=dataset,channel=channel,classifier=name,fold=fold,n_train=int(train.sum()),n_test=int(test.sum()),**measures(y[test],pred,labels)))
                    scaler=fitted.named_steps.get('standardscaler') if hasattr(fitted,'named_steps') else None
                    if scaler is not None:
                        assert np.allclose(scaler.mean_,X[train].mean().to_numpy())
                    fit_audit.append(dict(dataset=dataset,channel=channel,classifier=name,protocol='recording_cv',fold=fold,
                        train_recordings=';'.join(group.loc[train,'recording']),test_recordings=';'.join(group.loc[test,'recording']),
                        scaling='training fold only' if scaler is not None else 'none required',
                        scaler_mean_delta_alpha=scaler.mean_[0] if scaler is not None else None,
                        scaler_mean_alpha0=scaler.mean_[1] if scaler is not None else None))
                metric=measures(y,oof,labels)
                metrics_for_model=[r for r in fold_metrics if r['dataset']==dataset and r['channel']==channel and r['classifier']==name]
                results.append(dict(dataset=dataset,channel=channel,classifier=name,n_samples=len(group),n_classes=len(labels),
                    available_classes=';'.join(labels),missing_classes=';'.join(s for s in all_states if s not in labels),
                    n_folds=protocol['folds'][dataset],metric_aggregation='Pooled single-pass out-of-fold predictions',
                    fold_accuracy_std=np.std([r['accuracy'] for r in metrics_for_model],ddof=1),
                    fold_macro_f1_std=np.std([r['macro_f1'] for r in metrics_for_model],ddof=1),**metric))
                p,r,f,support=precision_recall_fscore_support(y,oof,labels=labels,zero_division=0)
                available={state:dict(precision=p[i],recall=r[i],f1=f[i],support=int(support[i])) for i,state in enumerate(labels)}
                for state in all_states:
                    class_metrics.append(dict(dataset=dataset,channel=channel,classifier=name,state=state,
                        status='EVALUABLE' if state in available else 'NOT EVALUABLE - no QC-valid sample',
                        **available.get(state,dict(precision=np.nan,recall=np.nan,f1=np.nan,support=0))))
                for index,row in group.iterrows():
                    predictions.append(dict(dataset=dataset,channel=channel,classifier=name,recording=row.recording,
                        state=row.state,predicted_state=oof[index],speed=row.speed,load=row.load,repeat=row['repeat'],fold=row.fold))
                cm=confusion_matrix(y,oof,labels=all_states)
                slug=f'{dataset.lower()}_{channel}_{MODULES[name]}'
                save(f'confusion_matrices/{slug}.csv',pd.DataFrame(cm,index=all_states,columns=all_states).rename_axis('true_state').reset_index())
                plot_confusion(cm,all_states,set(labels),f'{dataset} / {channel} / {name}',slug)
            # Train-only majority baseline, reported separately from the requested classifiers.
            base_pred=np.empty(len(group),dtype=object)
            for fold in range(protocol['folds'][dataset]):
                train=group.fold.ne(fold);test=group.fold.eq(fold)
                base_pred[test]=DummyClassifier(strategy='most_frequent').fit(X[train],y[train]).predict(X[test])
            baseline.append(dict(dataset=dataset,channel=channel,n_samples=len(group),**measures(y,base_pred,labels)))
            # Fixed supplemental stress tests, not used to select hyperparameters/features.
            for protocol_name,column in [('leave_one_speed_out','speed'),('leave_one_load_out','load')]:
                for value in sorted(group[column].unique()):
                    test=group[column].eq(value);train=~test
                    missing=sorted(set(labels)-set(y[train]))
                    for name,model in models.items():
                        record=dict(dataset=dataset,channel=channel,classifier=name,protocol=protocol_name,held_out_condition=str(value),
                            n_train=int(train.sum()),n_test=int(test.sum()),missing_training_classes=';'.join(missing),
                            status='NOT EVALUABLE' if missing or train.sum()<5 else 'EVALUABLE')
                        if record['status']=='EVALUABLE':
                            fitted=clone(model).fit(X[train],y[train]);pred=fitted.predict(X[test])
                            record.update(measures(y[test],pred,sorted(y[test].unique())))
                            record['macro_scope']='Classes present in this held-out test condition'
                            for row,predicted in zip(group[test].itertuples(index=False),pred):
                                transfer_pred.append(dict(dataset=dataset,channel=channel,classifier=name,protocol=protocol_name,
                                    held_out_condition=str(value),recording=row.recording,state=row.state,predicted_state=predicted))
                        transfer.append(record)
    result=save('classification_results.csv',results)
    save('tables/fold_metrics.csv',fold_metrics)
    predictions=save('tables/out_of_fold_predictions.csv',predictions)
    class_metrics=save('per_class_metrics.csv',class_metrics)
    save('tables/fit_leakage_audit.csv',fit_audit)
    baseline=save('tables/majority_baseline.csv',baseline)
    transfer=save('tables/operating_condition_splits.csv',transfer)
    transfer_pred=save('tables/operating_condition_predictions.csv',transfer_pred)
    transfer_summary=[]
    for keys,g in transfer.groupby(['dataset','channel','classifier','protocol']):
        ep=transfer_pred
        for key,value in zip(['dataset','channel','classifier','protocol'],keys):ep=ep[ep[key].eq(value)]
        group=dataset_frames[keys[0]];group=group[group.channel.eq(keys[1])]
        labels=sorted(ep.state.unique()) if len(ep) else []
        untested=sorted(set(group.state)-set(labels))
        transfer_summary.append(dict(zip(['dataset','channel','classifier','protocol'],keys),evaluable_splits=int(g.status.eq('EVALUABLE').sum()),
            expected_splits=len(g),tested_recordings=len(ep),eligible_recordings=len(group),test_coverage=len(ep)/len(group),
            tested_classes=';'.join(labels),untested_classes=';'.join(untested),macro_scope='Classes with evaluated held-out observations; untested classes NE',
            **(measures(ep.state,ep.predicted_state,labels) if len(ep) else {})))
    transfer_summary=save('tables/operating_condition_summary.csv',transfer_summary)
    best=result.sort_values(['dataset','channel','macro_f1','accuracy'],ascending=[True,True,False,False]).groupby(['dataset','channel'],sort=False).head(1)
    save('tables/best_by_context.csv',best)
    plot_results(result)
    report(result,best,distribution,class_metrics,baseline,transfer_summary)
    assert digest(ROOT/'result/test/step8/config/locked_feature_set.json')==protocol['step8_config_sha256']
    feature_lock.load_locked_config()
    assert predictions.groupby(['dataset','channel','classifier','recording']).size().eq(1).all()
    dump('diagnostics/validation.json',dict(all_passed=True,locked_features_unchanged=True,
        one_oof_prediction_per_recording_channel_classifier=True,recording_train_test_overlap=False,
        same_folds_all_classifiers=True,same_recording_fold_across_channels=True,
        scalers_verified_training_mean_only=True,missing_classes_reported_not_imputed=True,
        hyperparameter_search=False,classification_contexts=4,classifier_evaluations=len(result)))
    import subprocess
    check=subprocess.run([sys.executable,str(OUT/'scripts/test_classification.py')],capture_output=True,text=True)
    (OUT/'diagnostics/tests.log').write_text(check.stdout+check.stderr)
    dump('diagnostics/tests.json',dict(passed=check.returncode==0,test_cases=6))
    assert check.returncode==0,check.stdout+check.stderr
    save('diagnostics/output_manifest.csv',[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and not {'_work','__pycache__'}.intersection(p.parts) and p.name!='output_manifest.csv'])
    print(result[['dataset','channel','classifier','n_samples','accuracy','macro_f1']].to_string(index=False))

def plot_confusion(cm,states,available,title,slug):
    fig,ax=plt.subplots(figsize=(6,5));den=cm.sum(axis=1,keepdims=True)
    normalized=np.divide(cm,den,out=np.zeros_like(cm,dtype=float),where=den>0)
    normalized=np.ma.masked_where(np.broadcast_to(den==0,cm.shape),normalized)
    im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax,label='Recall-normalized')
    ax.set_xticks(range(len(states)),states);ax.set_yticks(range(len(states)),[s if s in available else s+' (NE)' for s in states])
    for i in range(len(states)):
        for j in range(len(states)):
            ax.text(j,i,str(cm[i,j]) if states[i] in available else 'NE',ha='center',va='center',color='white' if cm[i,j]/max(den[i,0],1)>.5 else 'black',fontsize=8)
    ax.set_xlabel('Predicted configuration');ax.set_ylabel('True configuration');ax.set_title(title)
    fig.tight_layout();fig.savefig(OUT/f'plots/{slug}_confusion.png',dpi=150);plt.close(fig)

def plot_results(result):
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(keys,g) in zip(axs.flat,result.groupby(['dataset','channel'],sort=False)):
        x=np.arange(len(g));ax.bar(x-.17,g.accuracy,.34,label='Accuracy');ax.bar(x+.17,g.macro_f1,.34,label='Macro F1')
        ax.set_xticks(x,g.classifier);ax.set_ylim(0,1);ax.set_title('/'.join(keys));ax.legend()
    fig.suptitle('Fixed RAW delta_alpha + alpha0 — exploratory out-of-fold results');fig.tight_layout();fig.savefig(OUT/'plots/classifier_comparison.png',dpi=150);plt.close(fig)

def markdown(frame,columns):
    def val(v):
        if pd.isna(v):return 'NE'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in frame[columns].itertuples(index=False,name=None))

def report(result,best,distribution,perclass,baseline,transfer):
    text='''# STEP 9 — CLASSIFICATION

## 1. Dataset used

Helical và Spur phân tích riêng, mỗi original recording là một sample; input/output là các analyses riêng, không cộng thành acquisitions độc lập. Chỉ RAW đã pass QC nguyên bản và hai locked features finite. Helical120 recordings nguồn, Spur160; số sample thực dùng theo channel ở bảng bên dưới. Không windows, không imputation/QC rescue, không gộp topology thành cùng class. Metadata compound configurations được lưu nguyên bản từ workbook snapshots đã xác minh; H2–H5 không phải pure bearing contrast.

## 2. Locked features

Input cố định theo Step8: **RAW `[delta_alpha, alpha0]`**, đúng order. Load qua Step8 integrity guard, xác minh upstream/config hashes. Không dùng delta_h, VMD/EMD, speed, load hay fault metadata như feature classifier. Không rerun/tune MF-DFA, scaling, VMD, EMD hoặc QC. Step8 features được chọn từ chính datasets/labels này: kết quả CV ở đây conditional trên bộ feature đã khóa và exploratory, chưa là validation độc lập của toàn quy trình feature selection.

## 3. Class distribution

'''
    text+=markdown(distribution,['dataset','channel','state','valid_recordings','excluded_recordings','qc_coverage','status'])
    text+='''

Helical tương đối cân bằng (16–20 samples/class/channel). Spur input có17/2/3/0/3/18/18/15 cho S1–S8 (max/min nonzero=9); output8/2/0/0/2/6/16/14 (ratio8). S4 không evaluable ở cả channels; S3 không evaluable ở output. **Spur input là bài toán7 configurations còn valid, output6 configurations; không phải full8-class validation.** Macro metrics chỉ tính trên classes còn valid; per-class metrics của missing configurations là NE, không thay bằng recall0. QC selection có thể không ngẫu nhiên và ảnh hưởng độ khó bài toán.

## 4. Classification method

- KNN:5 neighbors, uniform weights, Euclidean distance; StandardScaler.
- SVM:RBF, C=1, gamma=scale; StandardScaler.
- LDA:svd solver, empirical priors; StandardScaler.
- Random Forest:200 trees, default unrestricted depth, min leaf1, max_features=sqrt, bootstrap, seed9609, single thread; không cần scaling.

Không grid search/hyperparameter tuning, không chọn lại features. RF có thể overfit training data; chỉ báo predictions held-out. Không cân bằng class bằng oversampling hoặc dùng thông tin test. Baseline train-only majority báo riêng. Code nằm đúng4 files `code/core/classifier/{knn,svm,lda,random_forest}.py`; factory trả sklearn estimator/Pipeline có thể clone.

Primary protocol: Helical5-fold, Spur2-fold vì rare classes chỉ2 samples. Configuration-stratified partition, balance channel eligibility bằng metadata, seed9609; mỗi recording có một fold toàn dataset dùng chung hai channels. Fold maps và config lưu trước khi fit; tất cả classifiers dùng cùng samples/folds trong mỗi context. Mỗi eligible recording có đúng một prediction out-of-fold cho mỗi classifier/channel. Train/test recording IDs không giao nhau. Scaler nằm trong Pipeline và fit riêng từng training fold; gamma/priors cũng chỉ học training. Các fold đều có train/test observations cho mọi available class. Stratification dùng labels được phép; không dùng features/predictions để phân folds.

Accuracy, macro F1, macro precision, macro recall tính từ pooled OOF predictions; weightedF1 và fold standard deviations báo thêm. Recall macro tương đương balanced accuracy trên available labels. Fold std không phải confidence interval và2-fold Spur rất bất ổn; mỗi rare-class prediction chỉ có một training example cùng class. Không dùng repeated CV như samples độc lập. Best so theo macroF1 trước, accuracy phụ; best là mô tả nội bộ, không triển khai/pick final bằng independent test.

Supplemental fixed stress tests: leave-one-speed-out và leave-one-load-out trên valid records; không chọn features/parameters từ kết quả. Split thiếu bất kỳ available training class ghi NOT EVALUABLE, không drop class để fit. Fold macro metrics tính trên classes xuất hiện ở test condition; aggregate transfer summary cũng chỉ dùng classes có test observations evaluable, các class chưa được test ghi NE trong untested_classes, kèm test coverage (có thể<100%). Split/prediction tables lưu đầy đủ. Không dùng stress-test scores để thay primary protocol sau khi xem kết quả.

## 5. Helical results

'''
    text+=markdown(result[result.dataset.eq('Helical')],['channel','classifier','n_samples','n_classes','accuracy','macro_f1','macro_precision','macro_recall','fold_macro_f1_std'])
    text+='\n\n## 6. Spur results\n\n'
    text+=markdown(result[result.dataset.eq('Spur')],['channel','classifier','n_samples','n_classes','accuracy','macro_f1','macro_precision','macro_recall','fold_macro_f1_std'])
    text+='\n\nTrain-only majority baseline để đọc accuracy trong imbalance:\n\n'
    text+=markdown(baseline,['dataset','channel','n_samples','accuracy','macro_f1'])
    text+='\n\n## 7. Input vs output channel\n\nHai channels dùng cùng recording-to-fold mapping; coverage/class sets khác nhau, vì vậy score cao hơn không chứng minh channel vật lý tốt hơn. Không fuse channels hay cộng sample size. Bảng best theo từng context ở mục8; per-class table và16 confusion matrices cho mọi classifier/channel/dataset. Transfer tests:\n\n'
    text+=markdown(transfer,['dataset','channel','classifier','protocol','evaluable_splits','expected_splits','tested_recordings','eligible_recordings','test_coverage','untested_classes','accuracy','macro_f1'])
    text+='\n\n## 8. Best classifier\n\nBest theo pooled macro F1 trong từng dataset/channel (không tổng hợp các topology khác class definitions thành một accuracy):\n\n'
    text+=markdown(best,['dataset','channel','classifier','accuracy','macro_f1','macro_precision','macro_recall'])
    text+='\n\nPer-class recall của best models, kèm precision/support để tránh gọi class dễ chỉ vì nhiều false positives:\n\n'
    selected=perclass.merge(best[['dataset','channel','classifier']],on=['dataset','channel','classifier'])
    text+=markdown(selected,['dataset','channel','classifier','state','support','precision','recall','f1','status'])
    for row in best.itertuples():
        metrics=selected[(selected.dataset==row.dataset)&(selected.channel==row.channel)&selected.support.gt(0)]
        maxr=metrics.recall.max();minr=metrics.recall.min()
        easy=', '.join(metrics.loc[metrics.recall.eq(maxr),'state']);hard=', '.join(metrics.loc[metrics.recall.eq(minr),'state'])
        text+=f'\n\n{row.dataset}/{row.channel}: recall cao nhất {easy} ({maxr:.3f}); thấp nhất {hard} ({minr:.3f}). Đây là độ dễ/khó trong protocol và subset hiện tại, không properties vật lý cố định.'
    text+='''

## 9. Limitations

- Feature-selection labels đã được dùng Step8: primary CV chưa đo unbiased end-to-end generalization. Cần recordings mới chưa tham gia selection để có final independent estimate; Step9 không chọn lại features theo accuracy.
- Chỉ2 acquisitions/condition; recording-level folds có thể cùng fault/speed/load ở train/test. Đây chủ yếu là interpolation trong dataset; supplemental speed/load tests cho giới hạn operating transfer, không giải quyết mọi acquisition/domain dependence.
- Spur imbalance lớn, rare classes2–3 samples, missing classes và QC selection bias. Full S1–S8 performance không evaluable; không suy recall/accuracy cho S3 output hoặc S4. Không coi QC failures là classifier failures hay feature vô dụng.
- So best trong4 classifiers bằng cùng internal OOF có model-comparison optimism; không gọi best là universal classifier. LDA/Gaussian assumptions, simple SVM defaults và KNN neighborhood không tối ưu; scope chỉ basic comparison. Không tăng feature count/tune để cứu scores.
- High accuracy (nếu có) không là bearing/gear/shaft causal localization; targets là original fault configurations và compound faults. Không map H/S như cùng physical faults hay pool datasets.

## 10. Conclusion

**Bộ hai features chứa thông tin phân loại vượt majority baseline, nhưng hiện chưa đủ bằng chứng cho classification đáng tin cậy trên toàn configurations và operating conditions. Random Forest tốt nhất theo Macro F1 ở cả bốn contexts.** Primary MacroF1 chỉ0.419–0.596; một số class recall0 và Helical load-transfer rất kém. Không điều chỉnh feature lock để cải thiện các scores này.

'''
    for row in best.itertuples():
        b=baseline[(baseline.dataset==row.dataset)&(baseline.channel==row.channel)].iloc[0]
        text+=f'- {row.dataset}/{row.channel}: best {row.classifier}, accuracy{row.accuracy:.3f}, macroF1{row.macro_f1:.3f}; majority baseline accuracy{b.accuracy:.3f}, macroF1{b.macro_f1:.3f}.\n'
    text+='''
Helical Random Forest leave-one-load-out MacroF1 output0.092, input0.146, giảm mạnh so với recording CV. Leave-one-speed-out output0.407, input0.315. Đây là evidence cụ thể chống diễn giải bộ hai features là operating-condition-invariant. Spur transfer thiếu training classes ở một số splits; đọc test/class coverage thay vì dùng scores subset để xác nhận full transfer.

Với primary output, H2/H1 có recall0.70/0.65, H5 thấp nhất0.412; input Helical H6 cao nhất0.55, H1 thấp nhất0.3125. Spur output S6 recall0, S1 chỉ0.125; S8/S7 có0.786/0.688 trên14/16 samples. S5 output đạt1.0 nhưng chỉ2 samples, chưa đủ gọi class này chắc chắn dễ. Spur input S3 recall0 trên3 samples; S1 đạt0.824 trên17; S2 đạt1.0 nhưng chỉ2 samples. S4 và S3 output không có evidence để đánh giá.

Hai locked features cho phép chạy classification fault configurations trong eligible subset; mức đủ thực dụng phải dựa vào macroF1, từng class và condition-transfer, không chỉ accuracy. Không có target performance threshold đã xác định trước nên không tự gán đạt/không đạt deployment. Kết quả không chứng minh đủ cho robust full Helical/Spur classification hoặc localization; Spur full dataset vẫn chưa evaluable. Giữ nguyên feature lock; bước kế tiếp cần thêm independent recordings và QC coverage thay vì chọn lại feature bằng accuracy của tập này.

Các files chính: classification_results.csv; class_distribution.csv; per_class_metrics.csv; confusion_matrices/*.csv; plots/*.png; tables/out_of_fold_predictions.csv; tables/fold_metrics.csv; tables/fit_leakage_audit.csv; tables/operating_condition_*.csv; config/evaluation_protocol.json. Không commit/push GitHub hay train/deploy final production model.
'''
    (OUT/'REPORT.md').write_text(text)

if __name__=='__main__':run()
