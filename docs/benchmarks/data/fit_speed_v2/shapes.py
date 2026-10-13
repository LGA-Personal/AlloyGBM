import numpy as np
from sklearn.datasets import make_classification, make_regression, make_friedman1
def _mk(name):
    rng=np.random.default_rng(7)
    if name=='reg_60k_x18':
        X,y=make_friedman1(n_samples=60000,n_features=18,noise=1.0,random_state=1); return ('regression',X,y)
    if name=='bin_40k_x8':
        X,y=make_classification(n_samples=40000,n_features=8,n_informative=6,n_redundant=0,flip_y=0.05,random_state=2); return ('binary',X,y)
    if name=='mc10_1800_x216':
        X,y=make_classification(n_samples=1800,n_features=216,n_informative=40,n_classes=10,n_clusters_per_class=1,random_state=3); return ('multiclass',X,y)
    if name=='bin_3k_x1000_sparse':
        X,y=make_classification(n_samples=3000,n_features=1000,n_informative=20,flip_y=0.02,random_state=4)
        X[:, 30:] = (rng.random((3000,970))<0.02).astype(float); return ('binary',X,y)
    if name=='bin_900_x20_lowcard':
        X,y=make_classification(n_samples=900,n_features=20,n_informative=6,flip_y=0.1,random_state=5)
        X[:, :12] = np.floor(np.abs(X[:, :12])*2); return ('binary',X,y)
    if name=='reg_800_x17':
        X,y=make_regression(n_samples=800,n_features=17,n_informative=8,noise=10,random_state=6); return ('regression',X,y*100)
    if name=='mc26_7k_x300':
        X,y=make_classification(n_samples=7000,n_features=300,n_informative=50,n_classes=26,n_clusters_per_class=1,random_state=8); return ('multiclass',X,y)
    raise KeyError(name)
NAMES=['reg_60k_x18','bin_40k_x8','mc10_1800_x216','bin_3k_x1000_sparse','bin_900_x20_lowcard','reg_800_x17','mc26_7k_x300']
class _Lazy(dict):
    def __getitem__(self,k): return _mk(k)
    def items(self): return ((n,_mk(n)) for n in NAMES)
def shapes(): return _Lazy()
