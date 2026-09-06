exec(open('/tmp/alloygbm-review-20260906/accuracy_probe.py').read().split("out=Path(")[0])
from benchmarks.competitiveness.datasets import build_dataset_cases, fingerprint_case
out=Path('/tmp/alloygbm-review-20260906/confirm_accuracy.jsonl')
for seed in (20260902,20260903,20260904):
    deep.SEED=seed
    X,y,Xt,yt=deep.make_dataset(500000,40)
    factories=deep.build_factories(200,8,1)
    factories['alloy_linear']=lambda: alloygbm.GBMRegressor(n_estimators=200,max_depth=8,learning_rate=.05,row_subsample=.8,col_subsample=.8,continuous_binning_max_bins=256,continuous_binning_strategy='linear',seed=seed,deterministic=True,n_jobs=1)
    if seed==20260903: factories=dict(reversed(list(factories.items())))
    for lib,factory in factories.items():
        m=factory(); start=time.perf_counter(); m.fit(X,y); elapsed=time.perf_counter()-start
        p=m.predict(Xt)
        tail=np.any(np.abs(Xt[:,:5])>2.65,axis=1)
        rec=dict(seed=seed,library=lib,fit_seconds=elapsed,train_rmse=deep.rmse(y,m.predict(X)),test_rmse=deep.rmse(yt,p),tail_rows=int(tail.sum()),tail_rmse=deep.rmse(yt[tail],p[tail]),central_rmse=deep.rmse(yt[~tail],p[~tail]),timing_kind='single fit per seed; diagnostic, not speed gate')
        if hasattr(m,'get_all_params'):rec['resolved_catboost_params']=m.get_all_params()
        if hasattr(m,'resolved_training_policy_'):rec['resolved']=m.resolved_training_policy_
        with out.open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps({k:v for k,v in rec.items() if k not in ('resolved_catboost_params','resolved')}),flush=True)
