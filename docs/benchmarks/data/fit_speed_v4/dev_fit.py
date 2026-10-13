"""Default-fit test loss + fit time on the dev suite (never OpenML CC18/CTR23). Usage: acc.py OUT.json task_ids..."""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import rounds_cost as rc, dev_suite, frontier_matrix as fm
out=Path(sys.argv[1]); tids=[int(t) for t in sys.argv[2:]]
recs=json.loads(out.read_text()) if out.exists() else []
done={(r['dataset'],r['split']) for r in recs}
dss=dev_suite.load(rc.CACHE, splits=3, tasks=tids, path=rc.HERE/'dev_suite_manifest.json', log=lambda *_:None)
from alloygbm import GBMClassifier, GBMRegressor
for ds in sorted(dss,key=lambda d:len(d.y)):
    for s in range(3):
        if (ds.spec.name,s) in done: continue
        sp=fm.make_split(ds,s,None); Xa=fm._frame(ds.X,ds.categorical,'alloygbm')
        Xtr,Xte=fm._take(Xa,sp.train),fm._take(Xa,sp.test); ytr,yte=ds.y[sp.train],ds.y[sp.test]
        cls=GBMRegressor if ds.task==fm.REGRESSION else GBMClassifier
        m=cls(seed=s,n_jobs=4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); t=time.perf_counter(); m.fit(Xtr,ytr); dt=time.perf_counter()-t
        recs.append(dict(dataset=ds.spec.name,split=s,fit_seconds=dt,loss=rc.loss(ds.task,yte,rc.predict(m,ds.task,Xte))))
        out.write_text(json.dumps(recs,indent=1)); print(recs[-1],flush=True)
