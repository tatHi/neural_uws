import pickle
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np

# poisson
def poisson(k,lam=1):
    return (lam**(k)*np.exp(-lam))/math.factorial(k)

maxL = 8
data = [line.strip() for line in open('../../data/iphone_train_text.txt','r')]

# limit
data = data[:]

uniDict = defaultdict(lambda:0)    # 1文字の辞書

for line in tqdm(data):
    for i in range(len(line)):
        u = line[i]
        uniDict[u] += 1

# EOS
#uniDict['<BOS>'] = len(data)
#uniDict['<EOS>'] = len(data)

# count
uniAll = sum([uniDict[u] for u in uniDict])

uniProbDict = {}

for line in tqdm(data):
    for i in range(len(line)):
        for j in range(min(i+1,maxL)):
            w = line[i-j:i+1]
            if w not in uniProbDict:
                uniProbDict[w] = np.prod([uniDict[c]/uniAll for c in w]) * poisson(len(w))
# BOS EOS
uniProbDict['<BOS>'] = uniDict['<BOS>']/uniAll
uniProbDict['<EOS>'] = uniDict['<EOS>']/uniAll

pickle.dump(uniProbDict, open('../model/baseProb.dict', 'wb'))
