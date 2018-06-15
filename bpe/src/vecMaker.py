import lm as L
import dataset
from chainer import serializers
import pickle

import sys

d = sys.argv[1]

trainDataPath = '../data/%s_train_text_bpe.txt'%d
testDataPath = '../data/%s_test_text_bpe.txt'%d
ep = 29

data = [line.strip() for line in open(trainDataPath)]+[line.strip() for line in open(testDataPath)]
ds = dataset.Dataset(None, data=data,d=d)

lm = L.LM(d)
serializers.load_npz('../model/%s_lm_%d.npz'%(d,ep), lm)

print('calc voc vec')
Vs = list({w for line in ds.idData for w in line})
wvs = lm.getWordVecs(Vs).data

print('set dict')
vecDict = {w:wvs[i:i+1,] for i,w in enumerate(Vs)}

print('dumping')
pickle.dump(vecDict, open('../data/%s_wordVec_bpe.dict'%d, 'wb'))
