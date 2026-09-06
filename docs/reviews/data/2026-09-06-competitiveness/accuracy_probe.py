import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
for key in tuple(os.environ):
    if key.startswith('ALLOYGBM_'):
        del os.environ[key]
import sys,json,time,hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import alloygbm
from benchmarks import deep_scaling_comparison as deep
from benchmarks.auto_policy_benchmark import derive_candidate_params
out=Path('/tmp/alloygbm-review-20260906/accuracy_probe.jsonl')
common=dict(n_estimators=200,max_depth=8,learning_rate=0.05,row_subsample=0.8,col_subsample=0.8,continuous_binning_max_bins=256,seed=deep.SEED,deterministic=True,n_jobs=1)
X,y,Xt,yt=deep.make_dataset(500000,40)
base_policy=None
arms=['auto','no_gain_floor','manual_default','linear256','quantile1024','auto_repeat']
for arm in arms:
    params=dict(common)
    if arm in ('no_gain_floor','manual_default'):
        params.update(derive_candidate_params(arm,base_policy))
    elif arm=='linear256': params['continuous_binning_strategy']='linear'
    elif arm=='quantile1024': params['continuous_binning_max_bins']=1024
    m=alloygbm.GBMRegressor(**params)
    start=time.perf_counter(); m.fit(X,y); fit=time.perf_counter()-start
    pred=m.predict(Xt)
    rec=dict(arm=arm,fit_seconds=fit,train_rmse=deep.rmse(y,m.predict(X)),test_rmse=deep.rmse(yt,pred),rounds_completed=m.rounds_completed_,stop_reason=m.stop_reason_,resolved=m.resolved_training_policy_,params=params,fit_timing=m.fit_timing_,prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),artifact_sha256=hashlib.sha256(m.artifact_bytes).hexdigest(),module=alloygbm.__file__,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),host=platform.platform(),fixture=dict(rows=500000,features=40,seed=deep.SEED),timing_kind='diagnostic single fit; not performance acceptance evidence')
    if arm=='auto': base_policy=m.resolved_training_policy_
    with out.open('a') as f:f.write(json.dumps(rec)+'\n')
    print(json.dumps({k:rec[k] for k in ('arm','fit_seconds','train_rmse','test_rmse','rounds_completed','resolved')}),flush=True)
