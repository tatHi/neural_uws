import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F

# 文字ベクトルを事前学習
# CBOW

embedSize = 30
windowSize = 3 # 前後3文字

### MODEL ###
class CLM(Chain):
    #
    # 簡易CBOW
    # windowからtargetのindexを当てる

    def __init__(self, vocSize):
        super().__init__(
            embed = L.EmbedID(vocSize, embedSize),
            linear = L.Linear(embedSize, vocSize)
        )

    def getLoss(self, idLines):
        # idLine: [0,1,2,3,0]
        # one bos and one eos

        # BOS EOSを増やす
        idLines = [[idLine[0]]*(windowSize-1)+
                   idLine+
                   [idLine[-1]]*(windowSize-1) for idLine in idLines]

        prevs = []
        ts = []

        for idLine in idLines:
            vecs = self.embed(np.array(idLine,'i'))
            for t in range(windowSize, len(idLine)-windowSize):
                fvec = F.sum(vecs[t-windowSize:t,], axis=0)
                bvec = F.sum(vecs[t+1:t+1+windowSize,], axis=0)

                prevs.append(F.expand_dims(fvec+bvec, axis=0))

                ts.append(idLine[t])

        preds = self.linear(F.concat(prevs, axis=0))

        loss = F.softmax_cross_entropy(preds, np.array(ts,'i'))

        return loss

import dataset
import pickle
import module
from tqdm import tqdm
from chainer import optimizers

class Trainer:
    def __init__(self):
        data = [line.strip() for line in open('../../data/ntcir_train_text.txt') if line.strip()]
        self.ds = dataset.Dataset(data, init=True)
        pickle.dump((self.ds.char2id, self.ds.id2char), open('../model/ntcir/ids.dict','wb'))

        self.clm = CLM(len(self.ds.char2id))
        self.opt = optimizers.Adam()
        self.opt.setup(self.clm)

    def train(self, begin=0, end=20):
        for ep in range(begin, end):
            indices = np.random.permutation(len(self.ds.idData))
            batches = module.pack(indices, 64)

            accLoss = 0.
            
            for batch in tqdm(batches):
                self.clm.cleargrads()
                idLines = [self.ds.idData[b] for b in batch]
                loss = self.clm.getLoss(idLines)
                loss.backward()
                self.opt.update()
                
                accLoss += loss.data
            
            print(accLoss/len(batches))

            np.save('../model/ntcir/charEmbed.npy', self.clm.embed.W.data)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
