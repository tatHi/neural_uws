## mostly copied from model:mizuki.wordlm.py
import numpy as np
import chainer 
from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import Variable
from chainer.backends import cuda
import pickle
from time import time

maxL = 25#8
n = 3 # trigram
cSize = 3

embedSize = 30
wordSize = 100

wordVecMode = 'cai'

def prepareCNN(xs):
    # args: xs=[1,2,3,4,5]
    # return: [[1,2],[2,3],[3,4],[4,5]]
    ys = [xs[i:len(xs)-(n-i-2)] for i in range(n-1)]
    ys = F.concat(ys, axis=1)
    return ys

def ngramPair(x, size):
    # args: x=[1,2,3,4,5]
    # return: [[1,2,3],[2,3,4],[3,4,5]]
    # n-wordsに分ける
    ys = [tuple(x[i:i+size]) for i in range(len(x)-size+1)]
    return ys

class LM(Chain):
    def __init__(self, d):
        super().__init__(
            predLinear = L.Linear(wordSize*(n-1), wordSize),
            wordLinear = L.Linear(embedSize*maxL, wordSize)
        )

        self.charVecDict = np.load('../../uws/model/%s/charEmbed.npy'%d)

    def getWordVecs(self, words):
        ems = [F.reshape(self.charVecDict[list(word),],(embedSize*len(word),)) for word in words]
        ems = F.pad_sequence(ems, embedSize*maxL)
        wvs = F.tanh(self.wordLinear(ems))
        return wvs

    def getLoss(self, segIdLines, inVoc):
        # inVocは全語彙、真面目にsoftmaxとる
        ems = self.getWordVecs(inVoc)
       
        ems_lines = [F.concat([ems[[inVoc.index(seg)],] for seg in segIdLine[:-1]],axis=0)
                     for segIdLine in segIdLines]

        prevs = F.concat([F.concat([ems_line[:-1,], ems_line[1:,]],axis=1)
                          for ems_line in ems_lines],
                          axis=0)

        prevs = F.tanh(self.predLinear(prevs))
        
        # dot prev and ems
        dotTable = F.matmul(prevs, F.transpose(ems))

        # ts
        ts = [inVoc.index(seg) for segIdLine in segIdLines for seg in segIdLine[n-1:]]
        ts = np.array(ts, 'i')
        loss = F.softmax_cross_entropy(dotTable, ts)

        return loss

if __name__ == '__main__':
    lm = LM(10)
    lm.bowIndice=8
    lm.eowIndice=9
    line1 = [(0,),(0,),(1,2,3),(2,3,4),(2,3),(0,)]
    line2 = [(0,),(0,),(2,3,4,5),(1,),(0,)]
    voc = list(set(line1)|set(line2))
    print(line1)
    print(line2)
    print(voc)
    lm.cleargrads()
    import time
    st = time.time()
    loss = lm.getLoss([line1,line2], voc)
    loss.backward()
    print(time.time()-st)
