from chainer import functions as F
from chainer import Variable as V
import numpy as np
from time import time

st = time()

table = None
for _ in range(10000):
    a = np.random.rand(1,100).astype('f')

    if table is None:
        table = V(a)
    else:
        table = F.stack([table,a])

print(time()-st)
print(table.shape)
