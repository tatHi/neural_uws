from collections import defaultdict
import numpy as np
import pickle
import math
from tqdm import tqdm
bos = '<BOS>'
eos = '<EOS>'

class Dataset:
    def __init__(self, data, idDictPath=None):
        # param
        self.data = []
        self.id2char = {}
        self.char2id = {}
        self.cObs = Counter()
        self.idData = []
        self.segData = []
        if idDictPath is None:
            self.setDict(data)
        else:
            self.char2id, self.id2char = pickle.load(open(idDictPath,'rb'))
        self.setIdData(data)

        self.cW = None
        self.dist_p = None      # unigramでの分布
        self.dist_label = None  # 分布に対応するラベル(単語)

        # for limitation
        self.limitSize = None

    def setDict(self, data):
        countDict = defaultdict(lambda:0)
        for line in data:
            for c in line:
                countDict[c] += 1

        for k, v in reversed(sorted(countDict.items(), key=lambda x:x[1])):
            self.char2id[k] = len(self.char2id)
            self.id2char[self.char2id[k]] = k

        # bos eos
        for token in [bos, eos]:
            self.char2id[token] = len(self.char2id)
            self.id2char[self.char2id[token]] = token

    def setIdData(self, textPath):
        data = [line.strip() for line in open(textPath) if line.strip()]
        self.data = []
        self.idData = []
        self.segData = []
        for line in data:
            if len(line) == 0:
                continue
            self.data.append(line)
            self.idData.append([self.char2id[bos]]+self.chars2ids(line)+[self.char2id[eos]])
            self.segData.append([])

    def limitData(self, size):
        # limit data, idData, segData
        self.limitSize = size
        self.data = self.data[:size]
        self.idData = self.idData[:size]

    def setInitialSeg_random(self, setCObs=True):
        # uni
        self.segData = []
        maxL = 8
        for line in self.data:
            while True:
                segLine = [np.random.choice(2,1)[0] for _ in range(len(line)-1)]
                maxCount = 0
                count = 0
                for s in segLine:
                    if s==0:
                        count+=1
                        maxCount = max(maxCount, count)
                    else:   
                        count=0
                if maxCount < maxL:
                    break
            self.segData.append(segLine)


        if setCObs:
            for i in range(len(self.data)):
                # 初回のみEOSを追加
                self.cObs.add_sentence([bos]+self.getSegedLine(i)+[eos])

    def chars2ids(self, chars):
        # chars must be list of char
        if isinstance(chars, str):
            if chars==bos:
                return [self.char2id[bos]]
            elif chars==eos:
                return [self.char2id[eos]]
            else:
                chars = list(chars)
        ids = []
        for c in chars:
            ids.append(self.char2id[c])
        return ids

    def ids2chars(self, ids):
        chars = [self.id2char[wId] for wId in ids]
        return chars

    def reduceSeg(self, i):
        self.cW = None
        self.dist_p = None
        self.dist_label = None

        # 該当インデックスのsentenceに含まれる単語を削除
        line = self.data[i]
        segLine = self.segData[i]
        ws = []
        tmp = ''
        for s,c in zip(segLine, line):
            tmp += c
            if s == 1:
                ws.append(tmp)
                tmp = ''
        tmp += line[-1]
        ws.append(tmp)
        self.cObs.reduce_sentence(ws)

    def addSeg(self, i):
        self.cW = None
        self.dist_p = None
        self.dist_label = None

        # 該当インデックスのsentenceに含まれる単語を追加
        line = self.data[i]
        segLine = self.segData[i]
        ws = []
        tmp = ''
        for s,c in zip(segLine, line):
            tmp += c
            if s == 1:
                ws.append(tmp)
                tmp = ''
        tmp += line[-1]
        ws.append(tmp)
        self.cObs.add_sentence(ws)

    def getSegedLine(self, i):
        line = list(self.data[i])
        segs = self.segData[i]
        w = ''
        ws = []
        for seg, c in zip(segs, line):
            w+=c
            if seg==1:
                ws.append(w)
                w=''
        w+=line[-1]
        ws.append(w)
        return ws

    def getV(self):
        # count>0な語彙数
        V = len([w for w in self.cObs.uni if self.cObs.uni[w]>0])
        return V

    def getCW(self):
        if self.cW!=None:
            return self.cW
        cW = 0
        for w in self.cObs.uni:
            cW += self.cObs.get(w)
        self.cW = cW
        return self.cW

    def getWordProbs(self, words):
        # words is a list of strings
        cW = self.getCW()
        #ps = [(self.cObs.get(word)+1)/(cW+1) for word in words]
        ps = [(self.cObs.get(word))/(cW) for word in words]
        return ps

    def setDist(self, mode):

        self.dist_p = []
        self.dist_label = []

        for w in self.cObs.uni:
            if self.cObs.uni[w]>0:
                if mode=='noise':
                    # noise distribution
                    self.dist_p.append(self.cObs.uni[w]**(3/4))
                self.dist_label.append(w)

        # normalize
        self.dist_p = (self.dist_p/np.sum(self.dist_p)).tolist()

    def getInVoc(self, size=None, mode='uniform'):
        if size==None:
            vocs = [tuple(self.chars2ids(w)) for w in self.cObs.uni if self.cObs.get(w)>0]
        else:
            if mode=='noise':
                # noise distributionからサンプリング
                # size: 数
                if self.dist_label is None or self.dist_p is None:
                    self.setDist(mode)
                ws = np.random.choice(self.dist_label, size, p=self.dist_p, replace=False)
            elif mode=='uniform':
                # sample word from uniform distribution
                if self.dist_label is None:
                    self.setDist(mode)
                ws = np.random.choice(self.dist_label, size, replace=False)
            vocs = [tuple(self.chars2ids(w)) for w in ws]
        return vocs

    def negativeSampling(self, pos, size=10):
        # pos is tuple-ized word like (0,1,2)
        cand = self.getInVoc(size)
        if pos in cand:
            cand = self.negativeSampling(pos, size)
        return cand

    def getLambda(self):
        unk = 0
        W = 0
        for k in self.cObs.uni:
            w = self.cObs.get(k)
            W += w
            if w==1:
                unk += w
        
        lam = unk/W
        return lam

class Counter:
    def __init__(self):
        self.uni = {}

    def get(self, x):
        if x in self.uni:
            return self.uni[x]
        else:
            return 0

    def add(self, x):
        # uni
        if x in self.uni:
            self.uni[x] += 1
        else:
            self.uni[x] = 1

    def reduce(self, x):
        # uni
        if x in self.uni:
            self.uni[x] -= 1
        else:
            self.uni[x] = 0

        if self.uni[x] < 0:
            print('reduce error')
            exit()

    def add_sentence(self, segLine):
        # segLine: ['a', 'ab']
        for w in segLine:
            self.add(w)

    def reduce_sentence(self, segLine):
        # segLine: ['a', 'ab']
        for w in segLine:
            self.reduce(w)

