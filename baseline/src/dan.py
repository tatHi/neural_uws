# dan tagger
import numpy as np
import chainer
from chainer import Chain
from chainer import links as L
from chainer import functions as F
import pickle
import random

useChar = False
charSize = 30
wordSize = 100
hidSize = 100
classSize = 5

dropout = 0.2

class Classifier(Chain):
    def __init__(self):
        super().__init__(
            char2wordLinear = L.Linear(charSize, wordSize),
            linear1 = L.Linear(wordSize,hidSize),
            linear2 = L.Linear(hidSize,classSize)
        )

        self.wordVecDict = None
        self.charVecTable = None

    def setDict(self, wordVecDictPath=None):
        self.charVecTable = np.load('../../uws/model/charEmbed.npy') 
        if wordVecDictPath is not None:
            self.wordVecDict = pickle.load(open(wordVecDictPath, 'rb'))

    def forward(self, bows, ts=None):
        # bows: タプル形式の単語の塊
        # 学習済みの単語ベクトルと、文字ベクトルの写像でDAN

        testMode = ts is not None
        ys = []

        for bow in bows:
            # drop out
            if ts:
                while True:
                    neobow = [w for w in bow if random.random()>dropout]
                    if neobow:
                        bow = neobow
                        break

            N = 0
            y = chainer.Variable(np.zeros(wordSize).astype('f'))
            
            if useChar:
                # char embeding
                cs = [c for w in bow for c in w]
                cs_embed = F.tanh(self.char2wordLinear(self.charVecTable[cs,]))

                N += len(cs)
                y += F.sum(cs_embed, axis=0)

            # word embedding
            if self.wordVecDict is not None:
                N += len(bow)
                y += np.sum(np.concatenate([self.wordVecDict[w] for w in bow
                                            ],axis=0), axis=0)

            y /= N
            ys.append(y)

        ys = F.stack(ys)
        zs = self.linear2(self.linear1(ys))

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
