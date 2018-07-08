import sys
import segmentater
import dataset
import module
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mp','--modelPath',type=str,help='path where model path located')
parser.add_argument('-stp','--segedTextPath',type=str,help='path of segmentated text')
parser.add_argument('-bs','--batchSize',type=int,default=64,
                    help='batch size for vector calc.')

batchSize = 64
sg = None

ty = 'ntcir'

def setup():
    global sg
    sg = segmentater.Segmentater()
    sg.load(ep)

def train():
    # train
    indices = np.arange(len(sg.ds.idData))
    batches = module.pack(indices, batchSize)
    results_train = []
    for batch in tqdm(batches):
        results_train += sg.getOptSegmentation(batch) 

    # write
    f = open('../data/%s_train_text_uws%d.txt'%(ty,ep),'w')
    for line in results_train:
        f.write(line+'\n')
    f.close()

def test():
    # test
    data = [line.strip() for line in open('../../data/%s_test_text.txt'%ty) if line.strip()]
    sg.ds.setIdData(data)

    indices = np.arange(len(sg.ds.idData))
    batches = module.pack(indices, batchSize)
    results_test = []
    for batch in tqdm(batches):
        results_test += sg.getOptSegmentation(batch) 

    # write
    f = open('../data/%s_test_text_uws%d.txt'%(ty, ep),'w')
    for line in results_test:
        f.write(line+'\n')
    f.close()

def makeVecs():
    # set word vec
    ws = set()
    results_train = [line.strip() for line in open('../data/%s_train_text_uws%d.txt'%(ty,ep))]
    results_test = [line.strip() for line in open('../data/%s_test_text_uws%d.txt'%(ty,ep))]
    for line in results_train+results_test:
        line = line.split('　')
        ws |= set(line)
    ws = [tuple(sg.ds.chars2ids(w)) for w in list(ws)]
    ws += [(sg.ds.char2id[w],) for w in ['<BOS>','<EOS>']]

    # set
    sg.lm.getWordVecs(ws)

    wvDict = {}
    for w in ws:
        i = sg.lm.wordVecIndiceDict[w]
        wvDict[w] = sg.lm.wordVecTable[i:i+1,].data

    pickle.dump(wvDict, open('../data/%s_wordVec_uws%d.dict'%(ty,ep),'wb'))

#setup()
#train()
setup()
test()
setup()
makeVecs()
