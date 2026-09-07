exec(open('/tmp/alloygbm-review-20260906/accuracy_probe.py').read().split("out=Path(")[0])
X,y,Xt,yt=deep.make_dataset(500000,40)
out=Path('/tmp/alloygbm-review-20260906/quantile_budget_probe.jsonl')
for arm,denominator,threads in [('current_grid',256,1),('reserved_missing_grid',255,1),('reserved_missing_grid_t10',255,10)]:
    Xb=np.empty_like(X); Xbt=np.empty_like(Xt)
    for feature in range(X.shape[1]):
        values=np.sort(X[:,feature]); ranks=np.arange(1,denominator,dtype=np.int64)*len(values)//denominator
        cuts=np.unique(values[ranks])
        Xb[:,feature]=np.minimum(np.searchsorted(cuts,X[:,feature],side='right'),254)
        Xbt[:,feature]=np.minimum(np.searchsorted(cuts,Xt[:,feature],side='right'),254)
    m=alloygbm.GBMRegressor(n_estimators=200,max_depth=8,learning_rate=.05,row_subsample=.8,col_subsample=.8,continuous_binning_max_bins=256,seed=deep.SEED,deterministic=True,n_jobs=threads)
    start=time.perf_counter();m.fit(Xb,y);fit=time.perf_counter()-start
    p=m.predict(Xbt)
    rec=dict(arm=arm,n_jobs=threads,test_rmse=deep.rmse(yt,p),train_rmse=deep.rmse(y,m.predict(Xb)),fit_seconds_excluding_external_binning=fit,feature0_low_count=int((Xb[:,0]==0).sum()),feature0_top_count=int((Xb[:,0]==254).sum()),prediction_sha256=hashlib.sha256(p.tobytes()).hexdigest(),artifact_sha256=hashlib.sha256(m.artifact_bytes).hexdigest(),resolved=m.resolved_training_policy_,seed=deep.SEED,rows=500000,features=40,diagnostic='Externally pre-binned with existing auto trainer; timings exclude quantization, not comparable to raw fits')
    print(json.dumps(rec),flush=True)
    with out.open('a') as f:f.write(json.dumps(rec)+'\n')
