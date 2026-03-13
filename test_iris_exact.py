"""Test exact certification on Iris with higher beta for faster convergence."""
import os
os.environ['GRB_WLSACCESSID'] = '7999b455-6a85-40b2-aea2-1cd45f4c68ec'
os.environ['GRB_WLSSECRET'] = 'b9bbb913-2b9e-4a8c-bf60-4e1f0b1c5ded'
os.environ['GRB_LICENSEID'] = '2790219'

import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import optiml

iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)
y = np.eye(3)[iris.target]

print(f"Iris: n={X.shape[0]}, d={X.shape[1]}, outputs=3")
print(f"Testing exact=True (pure convex, no binary variables)\n")

model = optiml.ConvexReLUNet(in_features=4, out_features=3, n_patterns=200)

t0 = time.time()
model.fit(X, y, beta=0.1, tee=False, verbose=True, exact=True)
elapsed = time.time() - t0

print(f"\nCertified: {model.certified}")
print(f"Objective: {model.objective_value:.8f}")
print(f"Time: {elapsed:.1f}s")

pt = model.export('pytorch')
import torch
with torch.no_grad():
    preds = pt(torch.tensor(X, dtype=torch.float32)).numpy()
acc = (preds.argmax(1) == iris.target).mean() * 100
print(f"Train accuracy: {acc:.1f}%")
