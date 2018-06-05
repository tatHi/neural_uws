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
import charlm

maxL = 8
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
    def __init__(self, vocSize, id2char, charVecTablePath, uniProbDictPath):
        super().__init__(
            predLinear = L.Linear(wordSize*(n-1), wordSize),
            clm = charlm.CharLM(id2char, charVecTablePath, uniProbDictPath, vocSize=vocSize),
            wordLinear = L.Linear(embedSize*maxL, wordSize)
        )

        # 計算したことがあるパターンは無駄に計算しない
        #self.wordVecDict = {}
        self.ngramProbDict = {}

        self.wordVecTable = None
        self.wordVecIndiceDict = {}

        self.bowIndice = None
        self.eowIndice = None

        self.gpu = False

    def getWordVecs(self, words):
        # words: [(1,2,3), (2,3)]

        # wordVecDictに入ってない単語ベクトルを作る
        notInDict = [word for word in list(set(words)) if word not in self.wordVecIndiceDict]

        if notInDict:
            self.setWordVecs_cai(notInDict)

        # dictからベクトルを抜き出してwvsをつくる
        ids = [self.wordVecIndiceDict[word] for word in words]
        wvs = self.wordVecTable[ids,]
        return wvs

    def setWordVecs_cai(self, notInDict):
        #st = time()

        ems = [F.reshape(self.clm.getCharVecs(word),(embedSize*len(word),)) for word in notInDict]
        
        # padding
        ems = F.pad_sequence(ems, embedSize*maxL)

        # Linear
        wvs = F.tanh(self.wordLinear(ems))

        # set indice
        for i, word in enumerate(notInDict):
            self.wordVecIndiceDict[word] = len(self.wordVecIndiceDict)
        
        # set vecs
        if self.wordVecTable is None:
            self.wordVecTable = wvs
        else:
            self.wordVecTable = F.concat([self.wordVecTable, wvs], axis=0)
        #print('make word vec:', time()-st)

    def setNgramProbs(self, inVoc, V, pairs, ds):
        #alpha = 1
        #lam = alpha/(V+alpha)
        lam = ds.getLambda()

        # 計算されていないペアについて確率を求める
        pairs = list(set(pairs))
        notIn = [pair for pair in pairs \
                    if pair not in self.ngramProbDict]
         
        if not notIn:
            return

        # inVocに含まれていないtargetについてなら0
        if ds==None:
            # getLossから呼ばれるときは全部inVoc
            pass
        else:    
            oovs = []
            ivs = []
            notIn_n = [] # neural-trigramで計算するペア
            for pair in notIn:
                # targetがknown wordなら計算する
                #w = ''.join(ds.ids2chars(pair[-1]))
                #know = ds.cObs.get(w)>0

                know = True
                for w in pair:
                    if ds.cObs.get(''.join(ds.ids2chars(w)))==0:
                        know=False
                        break

                if know:
                    ivs.append(pair)
                else:
                    oovs.append(pair)
            
            st = time()
            self.setNgramProbs_OOV(oovs, lam)
            print('oov:', time()-st)
            
            st = time()
            self.setNgramProbs_IV(ivs, inVoc, V, lam)
            print('iv:', time()-st)

    def setNgramProbs_OOV(self, oovs, lam):
        for pair in oovs:
            p = lam*self.clm.getWordProb(pair[-1])
            self.ngramProbDict[pair] = p

    def setNgramProbs_IV(self, ivs, inVoc, V, lam):
        # count>0なtargetについてはニューラルで計算
        # 必要な単語のセットとベクトル
        notIn = ivs

        voc = list(set(inVoc) | set([w for pair in notIn for w in pair]))
        
        with chainer.no_backprop_mode():
            ems = self.getWordVecs(voc).data
        
        #st = time()
        vocIndDict = {w:_ for _,w in enumerate(voc)}
        notInIndices = [[vocIndDict[w] for w in pair] for pair in notIn]
        #print('make indice list:', time()-st)

        st = time()
        prevs = np.concatenate(
                    [ems[[nii[:-1]],].reshape(1,wordSize*(n-1))
                    for nii in notInIndices], axis=0
                )
        print('make prev vecs:',time()-st)

        st = time()
        preds = np.dot(prevs, self.predLinear.W.T.data)+self.predLinear.b.data
        print('linear:', time()-st)

        st = time()
        preds = np.tanh(preds)
        print('tanh:', time()-st)


        st = time()
        dotTable = np.dot(preds, ems.T)
        print('dot:', time()-st)

        # 各行の最大値を引いておく
        st = time()
        maxes = np.max(dotTable, axis=1)
        print('get max:', time()-st)
        
        st = time()
        expTable = dotTable - np.expand_dims(maxes, axis=1)
        print('dot-maxes(expanded_dims):',time()-st)

        # inVocの列だけ抜き出す
        st = time()
        expTable_inVoc = expTable[:,[vocIndDict[w] for w in inVoc]] 
        print('extraction:', time()-st)

        # inVocに対してexp
        st = time()
        expTable_inVoc = np.exp(expTable_inVoc)
        print('exp:', time()-st)

        # axis=1でsumとって正規化近似
        st = time()
        Zs = (V/len(inVoc))*np.sum(expTable_inVoc, axis=1)
        print('sum:', time()-st)

        # targetのexp
        ys = expTable[list(range(expTable.shape[0])),[vocIndDict[ni[-1]] for ni in notIn]]
        ys = np.exp(ys)

        # 分母Zsで割ってsoftmax近似
        ys /= Zs

        st = time()
        test=[]
        for ni,y in zip(notIn,ys):
            target = ni[-1]
            # 抜き出した項
            p = (1-lam)*y+\
                    lam*self.clm.getWordProb(target)
            self.ngramProbDict[ni] = p

        print('set dict:', time()-st)

#
    # 単一フレーズに対して確率を与える
    def getSentenceProb(self, segIdLine, inVoc, V, ds, prod=True):
        # arg: segIdLine = [(0,), (1,2), (3,4,5), (0,)]
        # inVoc is current words in vocabrary limited in the batch
        # V is current voc size in data

        pairs = ngramPair(segIdLine,n)

        # 未計算のngram確率をセット
        self.setNgramProbs(inVoc,V,pairs,ds)

        # dictからペアに対応する確率をとって積をとる
        ps = []
        for pair in pairs:
            p_ngram = self.ngramProbDict[pair]
            ps.append(p_ngram)

        '''
        # stack list of variable into variable
        print(ps)
        ps = F.stack(ps)
        '''

        if prod:
            return np.prod(ps)
        else:
            return ps

    def getSentenceLoss(self, segIdLine, inVoc, V):
        pairs = ngramPair(segIdLine,n)

        # 未計算のngram確率をセット
        self.setNgramProbs(inVoc,V,pairs,ds=None,forLoss=True)
       
        loss = 0
        for pair in pairs:
            s,z = self.ngramProbDict[pair]
            loss += (-s) + F.log(z)
        return loss/len(pairs)

    def getLoss_softmax(self, segIdLines, inVoc):
        # inVocは全語彙、真面目にsoftmaxとる
        
        # サンプリングで使ったベクトルを捨てる
        self.wordVecIndiceDict = {}
        self.wordVecTable = None

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
        if self.gpu:
            ts = cuda.to_gpu(ts)
        loss = F.softmax_cross_entropy(dotTable, ts)

        # clm
        words = [w for segIdLine in segIdLines for w in segIdLine]
        loss += self.clm.getLoss(words, self.gpu)

        # initialize
        self.wordVecTable = None
        self.wordVecIndiceDict = {}
        self.ngramProbDict = {}

        return loss

    def parameter_to(self, mode):
        if mode not in ['cpu', 'gpu']:
            print('lm.parameter_to:argument error', mode)
            exit()

        # charlm table
        if mode=='gpu':
            self.gpu = True
            self.clm.charVecTable = cuda.to_gpu(self.clm.charVecTable)
        else:
            self.gpu = False
            self.clm.charVecTable = cuda.to_cpu(self.clm.charVecTable)

        self.clm.embedTable = None
        self.clm.dot_table = None

        # lm wordVecDict
        self.wordVecTable = None
        self.wordVecIndiceDict = {}

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
    loss = lm.getLoss_softmax([line1,line2], voc)
    loss.backward()
    print(lm.clm.linear.W.grad)
    print(time.time()-st)
