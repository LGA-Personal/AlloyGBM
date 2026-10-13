import sys,time; sys.path.insert(0,'.')
from shapes import shapes
from alloygbm import GBMRegressor, GBMClassifier
name=sys.argv[1]; rounds=int(sys.argv[2])
task,X,y=shapes()[name]
E=GBMRegressor if task=='regression' else GBMClassifier
m=E(n_estimators=rounds,learning_rate=0.05,n_jobs=int(sys.argv[3]) if len(sys.argv)>3 else -1)
t=time.perf_counter(); m.fit(X,y); print(name, rounds, round(time.perf_counter()-t,3))
