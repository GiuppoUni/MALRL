import numpy as np
from sklearn.neighbors import KDTree
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
X = np.vstack( (X,np.array([0.1,0.1,0.1])) )
X = np.vstack( (X,np.array([-0.1,-0.1,-0.1])) )

# X = np.array(X)
print(X)
tree = KDTree(X, leaf_size=2)     # doctest: +SKIP
res =tree.query_radius([[0,0,0]], r=0.3, count_only=False)
print(res)
print([ X[i] for i in res])  # indices of neighbors within distance 0.3
