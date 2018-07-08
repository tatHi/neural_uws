import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
from chainer import optimizers, serializers
import lm as L
import dataset
import module
import pickle
import sampler
from tqdm import tqdm
from time import time

# 単語境界の前後2単語で確率を考える
n = 3-1
maxL = 8
batchSize = 8*4
preEp = -1

name = ''
k = 100

target = 'ntcir'

root = '../model/ntcir/'

textPath = '../../data/%s_train_text.txt'%target
idDictPath = '%sids.dict'%root
charVecTablePath = '%scharEmbed.npy'%root
uniProbDictPath = '%suniProb.dict'%root

def getPhrasePair(idLine, segLine, targetIndice):
    # idLine = [0,1,2,3,0]
    # segLine = [1,0,1,1]
    # paddingとそのsegTagが追加された状態で渡される
    
    ### targets ###
    for i in range(len(segLine)):
        if segLine[i]==1:
            if i < targetIndice:
                prev1 = i
            elif i > targetIndice:
                next1 = i
                break
    w0 = tuple(idLine[prev1+1:next1+1])
    w1L = tuple(idLine[prev1+1:targetIndice+1])
    w1R = tuple(idLine[targetIndice+1:next1+1])
   
    bos = tuple(idLine[:1])
    eos = tuple(idLine[-1:])
    idLine = idLine[1:-1]

    ### left words ###
    ls = []
    endL = prev1
    beginL = endL-1
    while beginL>=0 and len(ls)<n:
        if segLine[beginL] == 1 or beginL==0:
            ls.append(tuple(idLine[beginL:endL]))

            endL = beginL
            beginL = endL

        beginL -= 1

    # pad
    while len(ls) < n:
        ls.append(bos)

    ls = ls[::-1]

    ### right words ###
    rs = []
    beginR = next1
    endR = beginR+1
    while endR<len(segLine) and len(rs)<n:
        if segLine[endR] == 1 or endR == len(segLine):
            rs.append(tuple(idLine[beginR:endR]))
            beginR = endR
            endR = beginR
        endR += 1
    
    # pad
    # 文末なら一つだけeosをパディングする。
    # ifをwhileにするとbosと同じ数だけパディング（確率が壊れるのでやらない）
    if len(rs) < n:
        rs.append(eos)

    seg0 = ls + [w0] + rs
    seg1 = ls + [w1L, w1R] + rs

    return seg0, seg1

class Segmentater:
    def __init__(self):
        data = [line.strip() for line 
                    in open(textPath,'r') if line.strip()]
        self.ds = dataset.Dataset(data, idDictPath)

        self.ds.setInitialSeg_random()

        self.lm = L.LM(len(self.ds.char2id)+2, self.ds.id2char,
                        charVecTablePath, uniProbDictPath)
        self.lm.bowIndice = len(self.ds.char2id)
        self.lm.eowIndice = self.lm.bowIndice+1

        self.opt = optimizers.Adam()
        self.opt.setup(self.lm)

    def batchProcess(self, batch, pre=False):
        self.lm.cleargrads()

        if not pre:
            # reduce
            st = time()
            for b in batch:
                self.ds.reduceSeg(b)
            print('reduce:',time()-st)

            # inVoc and V
            st = time()
            V = len([w for w in self.ds.cObs.uni if self.ds.cObs.uni[w]>0])
            inVoc = self.ds.getInVoc(k,mode='uniform')
            print('countV and sample inVoc:', time()-st)

            # set lam
            alpha = 1
            lam = alpha/(V+alpha)

            # set ngram in batch
            st = time()
            sampler.setNgram([self.ds.idData[b][1:-1] for b in batch], self.lm, self.ds, inVoc)
            print('setNgram:',time()-st)

            st = time()
            for b in batch:
                print('_'.join(self.ds.getSegedLine(b)))
            
                self.ds.segData[b] = sampler.sample(self.ds.idData[b][1:-1], self.lm, self.ds)

                print('_'.join(self.ds.getSegedLine(b)))
            print('sampling:', time()-st)

            # add
            st = time()
            for b in batch:
                self.ds.addSeg(b)
            print('add:',time()-st)

        # recalc inVoc and V
        V = len([w for w in self.ds.cObs.uni if self.ds.cObs.uni[w]>0])
        
        # calc loss
        segedLines = [module.segmentIdLine(self.ds.idData[b], [1]+self.ds.segData[b]+[1])
                      for b in batch]
        # BOS Padding
        segedLines = [[segedLine[0] for _ in range(n-1)]+segedLine for segedLine in segedLines]

        st = time()
        
        loss = self.lm.getLoss(segedLines, self.ds.getInVoc())

        print('getLoss:',time()-st)

        print(loss)
        st = time()
        loss.backward()
        print('backward:',time()-st)

        st = time()
        self.opt.update()
        print('update:',time()-st)

    def save(self, ep):
        serializers.save_npz('%s%slm_%d.npz'%(root,name,ep), self.lm)
        pickle.dump(self.ds, open('%s%sds_%d.pickle'%(root,name,ep),'wb'))

    def load(self, ep):
        self.ds = pickle.load(open('%s%sds_%d.pickle'%(root,name,ep),'rb'))
        self.lm = L.LM(len(self.ds.char2id)+2, self.ds.id2char,
                                charVecTablePath, uniProbDictPath)
        serializers.load_npz('%s%slm_%d.npz'%(root,name,ep), self.lm)

    def train(self, begin=0, end=1000):
        if begin>0:
            self.load(begin-1)

        for ep in range(begin, end):
            print(ep)
            indices = np.random.permutation(len(self.ds.idData))
            batches = module.pack(indices, batchSize)
            for batch in tqdm(batches):
                self.batchProcess(batch, ep<preEp)
            self.save(ep)

    def getOptSegmentation(self, batch):
        inVoc = self.ds.getInVoc()
        idLines = [self.ds.idData[b] for b in batch]
        sampler.setNgram(idLines, self.lm, self.ds, inVoc)

        results = []
        for b in batch:
            idLine = self.ds.idData[b]
            #self.ds.segData[b] = sampler.viterbi(idLine[1:-1], self.lm, self.ds)
            self.ds.segData[b] = sampler.track(idLine[1:-1], self.lm, self.ds)

            results.append('　'.join(self.ds.getSegedLine(b)))
    
        return results

def showTempSegLine(begin=0, end=10, size=10):
    sg = Segmentater()
    for ep in range(begin, end):
        print(ep)
        sg.load(ep)

        for i in range(size):
            print('_'.join(sg.ds.getSegedLine(i)))

def showOptSegLine(begin=0, end=30, size=20):
    sg = Segmentater()
    for ep in range(begin, end):
        print(ep)
        sg.load(ep)
        batch = np.arange(size)
        results = sg.getOptSegmentation(batch)
        for r in results:
            print(r)

if __name__ == '__main__':
    sg = Segmentater()
    sg.train(0, 100)

