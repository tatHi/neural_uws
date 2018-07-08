import numpy as np
from chainer import Chain
from chainer import links as L
from chainer import functions as F

# 文字ベクトルを事前学習
# CBOW

np.random.seed(1234)

### MODEL ###
class CLM(Chain):
    #
    # 簡易CBOW
    # windowからtargetのindexを当てる

    def __init__(self, vocSize, embedSize, windowSize):
        super().__init__(
            embed = L.EmbedID(vocSize, embedSize),
            linear = L.Linear(embedSize, vocSize)
        )
        self.windowSize = windowSize

    def getLoss(self, idLines):
        # idLine: [0,1,2,3,0]
        # one bos and one eos

        # BOS EOSを増やす
        idLines = [[idLine[0]]*(self.windowSize-1)+
                   idLine+
                   [idLine[-1]]*(self.windowSize-1) for idLine in idLines]

        prevs = []
        ts = []

        for idLine in idLines:
            vecs = self.embed(np.array(idLine,'i'))
            for t in range(self.windowSize, len(idLine)-self.windowSize):
                fvec = F.sum(vecs[t-self.windowSize:t,], axis=0)
                bvec = F.sum(vecs[t+1:t+1+self.windowSize,], axis=0)

                prevs.append(F.expand_dims(fvec+bvec, axis=0))

                ts.append(idLine[t])

        preds = self.linear(F.concat(prevs, axis=0))

        loss = F.softmax_cross_entropy(preds, np.array(ts,'i'))

        return loss

import dataset
import pickle
import module
from chainer import optimizers
from time import time

class Trainer:
    def __init__(self, embedSize, windowSize, textPath, resultPath):
        idDictPath = resultPath+'/ids.dict'
        self.modelPath = resultPath+'/charEmbed.npy'

        data = [line.strip() for line in open(textPath) if line.strip()]
        self.ds = dataset.Dataset(data, init=True)
        pickle.dump((self.ds.char2id, self.ds.id2char), open(idDictPath,'wb'))

        self.clm = CLM(len(self.ds.char2id), embedSize, windowSize)
        self.opt = optimizers.Adam()
        self.opt.setup(self.clm)

    def train(self, maxEp, batchSize):
        for ep in range(1,maxEp+1):
            startTime = time()
            indices = np.random.permutation(len(self.ds.idData))
            batches = module.pack(indices, batchSize)

            accLoss = 0.
            
            for batch in batches:
                self.clm.cleargrads()
                idLines = [self.ds.idData[b] for b in batch]
                loss = self.clm.getLoss(idLines)
                loss.backward()
                self.opt.update()
                
                accLoss += loss.data
            
            print('epoch:%d\t loss:%f\ttime:%f'%(ep,accLoss/len(batches),time()-startTime))

            np.save(self.modelPath, self.clm.embed.W.data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp','--textPath',type=str,help='path of text to segmentation')
    parser.add_argument('-rp','--resultPath',type=str,default='../result',
                        help='path of result to save model and id-dictionary')
    parser.add_argument('-es','--embedSize',type=int,default=30,
                        help='charcter-embedding size')
    parser.add_argument('-ws','--windowSize',type=int,default=3,
                        help='window size for CBOW. train char-vec predicting target char from windowSize*2 words around.')
    parser.add_argument('-ep','--epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('-bs','--batchSize',type=int,default=64,
                        help='batch size for train')
    args = parser.parse_args()
    
    embedSize = args.embedSize
    windowSize = args.windowSize
    resultPath = args.resultPath

    trainer = Trainer(args.embedSize, args.windowSize, args.textPath, args.resultPath)
    trainer.train(args.epoch,args.batchSize)
