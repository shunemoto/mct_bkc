import numpy as np

X = np.array([(1.0, 2.0), (4.0, 3.0), (5.0, 7.0)], dtype=float)

diff = X[1:] - X[:-1]                 # 連続点の差分ベクトル (N-1, 2)
seg  = np.linalg.norm(diff, axis=1)   # 各区間の距離 (N-1,)

print(diff)
print(seg)