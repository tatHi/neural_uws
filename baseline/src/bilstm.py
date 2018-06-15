# dan tagger
import numpy as np
import chainer
from chainer import Chain
from chainer import links as L
from chainer import functions as F
import pickle
import random

wordSize = 100
hidSize = 100
classSize = 5

dropout = 0.2

class Classifier(Chain):
    def __init__(self):
        super().__init__(
            bilstm = L.NStepBiLSTM(1, wordSize, hidSize, dropout=0.2),
            linear = L.Linear(hidSize*2, classSize)
        )

        self.wordVecDict = None

    def setDict(self, wordVecDictPath):
        self.wordVecDict = pickle.load(open(wordVecDictPath, 'rb'))

    def forward(self, bows, ts=None):
        # bows: タプル形式の単語の塊
        # 学習済みの単語ベクトルと、文字ベクトルの写像でDAN

        testMode = ts is not None
        xs = []

        for bow in bows:
            # word embedding
            if self.wordVecDict is not None:
                x = np.concatenate([self.wordVecDict[w] for w in bow],axis=0)
            xs.append(x)

        hy,_,_ = self.bilstm(None, None, xs)

        hy = F.concat(F.separate(hy),axis=1)
        zs = self.linear(hy)

        if ts:
            ts = np.array(ts, 'i')
            return F.softmax_cross_entropy(zs, ts)
        else:
            return F.softmax(zs)

if __name__ == '__main__':
    c = Classifier()
    c.setDict()
    a = c.forward([[(1,2,3),(1,2)],[(2,),(1,2,3),(3,4)]])
    print(a)
