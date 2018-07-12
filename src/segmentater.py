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

n = 3-1

class Segmentater:
    def __init__(self,textPath, idDictPath, charVecPath, uniProbDictPath, resultPath, samplingSizeK):
        self.textPath = textPath
        self.idDictPath = idDictPath
        self.modelPath = resultPath+'/model.npz'
        self.datasetPath = resultPath+'/dataset.pickle'

        self.charVecPath = charVecPath
        self.uniProbDictPath = uniProbDictPath

        self.samplingSizeK = samplingSizeK

    def setupDS(self):
        self.ds = dataset.Dataset(self.textPath, self.idDictPath)

        # set initial segmentation
        self.ds.setInitialSeg_random()

    def setupLM(self):
        # language model
        self.lm = L.LM(len(self.ds.char2id)+2, self.ds.id2char,
                        charVecPath, uniProbDictPath)
        
        self.lm.bowIndice = len(self.ds.char2id)
        self.lm.eowIndice = self.lm.bowIndice+1

    def batchProcess(self, batch, showSeg):
        self.lm.cleargrads()

        # reduce
        for b in batch:
            self.ds.reduceSeg(b)

        # inVoc and V
        V = len([w for w in self.ds.cObs.uni if self.ds.cObs.uni[w]>0])
        inVoc = self.ds.getInVoc(self.samplingSizeK, mode='uniform')

        # set ngram in batch
        sampler.setNgram([self.ds.idData[b][1:-1] for b in batch], self.lm, self.ds, inVoc)

        for b in batch:
            if showSeg:
                print('red<','_'.join(self.ds.getSegedLine(b)))
        
            self.ds.segData[b] = sampler.sample(self.ds.idData[b][1:-1], self.lm, self.ds)

            if showSeg:
                print('add>','_'.join(self.ds.getSegedLine(b)))

        # add
        for b in batch:
            self.ds.addSeg(b)

        # recalc inVoc and V
        V = len([w for w in self.ds.cObs.uni if self.ds.cObs.uni[w]>0])
        
        # calc loss
        segedLines = [module.segmentIdLine(self.ds.idData[b], [1]+self.ds.segData[b]+[1])
                      for b in batch]
        # BOS Padding
        segedLines = [[segedLine[0] for _ in range(n-1)]+segedLine for segedLine in segedLines]

        loss = self.lm.getLoss(segedLines, self.ds.getInVoc())
        loss.backward()
        self.opt.update()

    def save(self):
        serializers.save_npz(self.modelPath, self.lm)
        pickle.dump(self.ds, open(self.datasetPath,'wb'))

    def load(self):
        self.ds = pickle.load(open(self.datasetPath,'rb'))
        self.lm = L.LM(len(self.ds.char2id)+2, self.ds.id2char,
                                self.charVecPath, self.uniProbDictPath)
        serializers.load_npz(self.modelPath, self.lm)

    def train(self, beginEpoch, endEpoch, batchSize, showSeg):
        if beginEpoch==0:
            self.setupDS()
            self.setupLM()
        else:
            self.load()
        self.opt = optimizers.Adam()
        self.opt.setup(self.lm)

        for ep in range(beginEpoch, endEpoch):
            indices = np.random.permutation(len(self.ds.idData))
            batches = module.pack(indices, batchSize)
            for i,batch in enumerate(batches):
                startTime = time()
                self.batchProcess(batch,showSeg)
                print('epoch:%d\tbatch:%d/%d\ttime:%f'%(ep, i+1, len(batches), time()-startTime))
            self.save()

    def segmentate(self, textPath, batchSize):
        self.load()

        # set text data
        self.ds.setIdData(textPath)

        batches = module.pack(np.arange(len(self.ds.idData)), batchSize)

        for batch in batches:
            inVoc = self.ds.getInVoc()
            idLines = [self.ds.idData[b][1:-1] for b in batch]
            sampler.setNgram(idLines, self.lm, self.ds, inVoc)

            for b in batch:
                idLine = self.ds.idData[b]
                self.ds.segData[b] = sampler.track(idLine[1:-1], self.lm, self.ds)

                yield '　'.join(self.ds.getSegedLine(b))    
    
    def assignVec(self, segedTextPath, resultPath):
        self.load()
        print('load model')

        # set word vec
        segedText = [line.strip() for line in open(segedTextPath)]
        ws = {w for line in segedText for w in line.split('　')}
        
        ws = [tuple(sg.ds.chars2ids(w)) for w in list(ws)]
        ws += [(sg.ds.char2id[w],) for w in ['<BOS>','<EOS>']]

        print('set words')

        # set
        sg.lm.getWordVecs(ws)
        print('calc vec')

        wvDict = {}
        for w in ws:
            i = sg.lm.wordVecIndiceDict[w]
            wvDict[w] = sg.lm.wordVecTable[i:i+1,].data

        writePath = resultPath+'/wordVec.dict'
        pickle.dump(wvDict, open(writePath,'wb'))
        print('done')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',type=str,choices=['train','seg','vecAssign'],
                        help='run mode')
    parser.add_argument('-tp','--textPath',type=str,help='text path for segmentation')
    parser.add_argument('-pp','--pretrainPath',type=str,
                        help='path where pretrained models and idDict are located')
    parser.add_argument('-rp','--resultPath',type=str,help='path to save models')

    parser.add_argument('-be','--beginEpoch',type=int,default=0,
                        help='train-begin epoch, if >0, existing model files are to be loaded')
    parser.add_argument('-ee','--endEpoch',type=int,default=50,
                        help='train-end epoch')
    parser.add_argument('-bs','--batchSize',type=int,default=32,
                        help='segmentation batch size')
    parser.add_argument('-ss','--showSeg',action='store_true',
                        help='show segmentation process')
    parser.add_argument('-K','--samplingSizeK',type=int,default=100,
                        help='sampling size k for softmax approximation')
    parser.add_argument('-stp','--segedTextPath',type=str,
                        help='segmentated text path')

    args = parser.parse_args()

    textPath = args.textPath
    idDictPath = '%s/ids.dict'%args.pretrainPath
    charVecPath = '%s/charEmbed.npy'%args.pretrainPath
    uniProbDictPath = '%s/uniProb.dict'%args.pretrainPath
    resultPath = args.resultPath

    sg = Segmentater(textPath, idDictPath, charVecPath, uniProbDictPath, resultPath, args.samplingSizeK)
    
    if args.mode == 'train':
        sg.train(args.beginEpoch, args.endEpoch, args.batchSize, args.showSeg)
    elif args.mode == 'seg':
        for line in sg.segmentate(args.textPath, args.batchSize):
            print(line)
    elif args.mode == 'vecAssign':
        sg.assignVec(args.segedTextPath, resultPath) 
