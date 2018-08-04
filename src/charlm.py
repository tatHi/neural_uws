import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Chain
from chainer.backends import cuda
import pickle

fixedEmbed = True

class CharLM(Chain):
    def __init__(self, id2char, charVecTablePath, uniProbDictPath, vocSize=None):
        super().__init__() 
        if fixedEmbed:
            self.charVecTable = np.load(charVecTablePath)
            size = self.charVecTable.shape[1]
        else:
            size = 30
            self.add_link('embed', L.EmbedID(vocSize, size))

        self.add_link('linear', L.Linear(size, size))

        self.dot_table = None
        self.softmax_table = None
        self.embedTable = None

        # uniProbDict
        self.id2char = id2char
        self.uniProbDict = pickle.load(open(uniProbDictPath, 'rb'))
        self.wordProbDict = {}

    def getCharVecs(self, ids):
        if self.embedTable is None:
            if fixedEmbed:
                self.embedTable = self.charVecTable
            else:
                self.embedTable = self.embed.W
        vecs = self.embedTable[[ids],]
        return vecs

    def setTable(self):
        # set two table
        if fixedEmbed:
            voc = self.charVecTable 
        else:
            voc = self.embed.W
        preds = self.linear(voc)
        self.dot_table = F.matmul(preds, F.transpose(voc))
        with chainer.no_backprop_mode():
            self.softmax_table = F.softmax(self.dot_table).data.astype(float) 

    def getWordProb(self, word):
        # word = [1,2,3]
        if self.softmax_table is None:
            self.setTable()

        if word not in self.wordProbDict:
            # p of word[0]
            p = self.uniProbDict[self.id2char[word[0]]]
            for i in range(len(word)-1):
                p *= self.softmax_table[word[i]][word[i+1]]
            self.wordProbDict[word] = p

        return self.wordProbDict[word]

    def getLoss(self, words, gpu):
        self.setTable()

        ids = [c for word in words for c in word[:-1]]
        rows = self.dot_table[ids,]

        ts = np.array([c for word in words for c in word[1:]], 'i')

        if gpu:
            ts = cuda.to_gpu(ts)

        loss = F.softmax_cross_entropy(rows, ts)

        # initialize dict
        self.wordProbDict = {}
        self.dot_table = None
        self.softmax_table = None
        if fixedEmbed==False:
            self.embedTable = None

        return loss

if __name__ == '__main__':
    clm = CharLM()
    import time
    clm.cleargrads()
    st = time.time()
    loss = clm.getLoss([[1,2,3],[2,3,4,5],[2,3,4,0,1]])
    print(loss)
    loss.backward()
    print(clm.embed.W.grad)
    print(clm.linear.W.grad)
    print(time.time()-st)
