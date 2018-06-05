# dp sampler
import numpy as np
from time import time
n = 3
maxL = 8

# trigram
def setNgram(idLines, lm, ds, inVoc):
    st = time()
    
    # idLines: 文字idのリスト*バッチサイズ
    # lm: NLM
    # ds: dataset(単語頻度の数え上げなどを行う)
    # inVoc: softmaxをとる時に考慮する語彙
    #        k個サンプリングし、batch内で同じものを使う。
    #        [(1,2,3),(2,3)]　文字のタプルで単語を表現したリスト

    # 何回も使うのでbos eosのidを保持しておく
    bos = (ds.char2id['<BOS>'],)
    eos = (ds.char2id['<EOS>'],)

    # 事前にngramペアを洗い出してlmのngramPairをセットしておく
    #   batch内に出現しうるn-gram（ここではtri-gram）
    #   tri-gramはtuple(tuple(c,c,c), tuple(c,c), tuple(c,c))という感じ
    #   idのタプルで表現された単語、のタプルでngram
    
    pairs = set()

    for idLine in idLines:
        # 文ごとに可能なペアを洗い出す
        
        for i in range(len(idLine)):
            # i文字目を末尾として、
            # target = w[i-j:i+1]
            # prev1 = w[i-j-k-1:i-j]
            # prev2 = w[i-j-k-l-2:i-j-k-1]
            # がtrigram=tuple(prev2, prev1, target)を構成する
            for j in range(min(i+1, maxL)):
                if i<len(idLine):
                    target = tuple(idLine[i-j:i+1])

                # prevs
                if i-j==0:
                    pairs.add((bos,bos,target))
                    # eos
                    if i==len(idLine)-1:
                        pairs.add((bos,target,eos))
                    continue

                for k in range(min(i-j, maxL)):
                    prev1 = tuple(idLine[i-j-k-1:i-j])
                    if i-j-k-1==0:
                        pairs.add((bos,prev1,target))
                    else:
                        for l in range(min(i-j-k-1, maxL)):
                            prev2 = tuple(idLine[i-j-k-l-2:i-j-k-1])
                            pairs.add((prev2,prev1,target))

                    # eos
                    if i==len(idLine)-1:
                        pairs.add((prev1,target,eos))

    pairs = list(pairs)
    print('\tmake voc set:',time()-st)

    # ngram pairについて確率をセット
    # NLMに以下の引数を投げて、求めたすべてのペアについて確率をNLM内のdictにセットする
    # lm.ngramProbDict[(pair)]=0.01という感じになる
    # inVoc: softmaxで考慮する語彙
    # ds.getV(): 現在の語彙数, int, softmax近似に使用する
    # ds: lm内で既知語か未知語かを判定するために渡す。単語頻度などを持ったクラス。
    lm.setNgramProbs(inVoc, ds.getV(), pairs, ds)        

def makeAlpha(idLine, lm, ds):
    # NLM内のdictに、batchで用いるtri-gramごとの確率をセットしてから使う
    # NLM内のdictにアクセスしてDPテーブルを作り、サンプリングする

    bos = (ds.char2id['<BOS>'],)
    eos = (ds.char2id['<EOS>'],)

    # dpテーブルを初期化
    alpha = np.array([[[0. for k in range(maxL)]\
                           for j in range(maxL)]\
                           for i in range(len(idLine))])
   
    # 遷移確率はNLMが持つdictにアクセスして取ってくる
    # (lm.ngramProbDict)
    # ngram pairそのものはもう一度つくる
    for t in range(len(idLine)):
        for k in range(min(t+1, maxL)):
            target = tuple(idLine[t-k:t+1])
           
            # prevs
            if t-k==0:
                pair = (bos,bos,target)
                alpha[t][k][0] = lm.ngramProbDict[pair]
                if t==len(idLine)-1:
                    # 末尾ならeos確率も掛け合わせる
                    pair = (bos,target,eos)
                    p = lm.ngramProbDict[pair]
                    alpha[t][k][0] *= p
                continue

            for j in range(min(t-k, maxL)):
                prev1 = tuple(idLine[t-k-j-1:t-k])
                if t-k-j-1==0:
                    pair = (bos,prev1,target)
                    alpha[t][k][j] = lm.ngramProbDict[pair]*alpha[t-k-1][j][0]
                else:
                    p = 0
                    for i in range(min(t-k-j-1, maxL)):
                        prev2 = tuple(idLine[t-k-j-i-2:t-k-j-1])
                        pair = (prev2,prev1,target)
                        p += lm.ngramProbDict[pair]*alpha[t-k-1][j][i]
                    alpha[t][k][j] = p
                    #print(alpha[t][k][j])

                if t==len(idLine)-1:
                    # 末尾ならeos確率も掛け合わせる
                    # (eosへの接続確率を末尾の単語のサンプリングに用いるため)
                    pair = (prev1,target,eos)
                    p = lm.ngramProbDict[pair]
                    alpha[t][k][j] *= p

    return alpha

def sample(idLine, lm, ds, tau=0.1):

    alpha = makeAlpha(idLine,lm,ds)

    # sample phase 1単語ずつサンプリングする
    #   alpha[t][k][j]はt文字目が末尾の時、t-k:t+1, t-k-j-1:t-kがともに単語である確率
    #   alpha[t]の平面について、横方向にsumを取ることで、次のkをサンプリングする。

    ls = []
    flag = len(idLine)-1
    while flag>=0:
        # dist of neural
        dist_n = np.sum(alpha[flag],axis=1)
        dist_n = dist_n/np.sum(dist_n)
        '''
        # dist of discrete
        dist_d = [ds.getWordProb(''.join(ds.ids2chars(idLine[flag-a:flag+1])))
                                            for a in range(min(flag+1, maxL))]
        dist_d = np.pad(dist_d, (0,maxL-len(dist_d)), 'constant', constant_values=0)
        dist_d = dist_d/np.sum(dist_d)

        dist = tau*dist_d + (1-tau)*dist_n
        '''
        dist = dist_n
        l = np.random.choice(len(dist), 1, p=dist)[0]
        
        ls.append(l+1)
        if sum(ls) == len(idLine):
            break
        flag -= l+1

    flag = 0
    segLine = []
    for l in ls[::-1]:
        flag += l
        segLine += [0 for _ in range(l-1)] + [1]
    segLine = segLine[:-1]
    return segLine

def track(idLine, lm, ds):
    alpha = makeAlpha(idLine, lm, ds)
    ls = []
    flag = len(idLine)-1
    while flag>=0:
        dist = np.sum(alpha[flag],axis=1)
        dist /= np.sum(dist)
        l = np.random.choice(len(dist), 1, p=dist)[0]
        ls.append(l+1)
        if sum(ls) == len(idLine):
            break
        flag -= l+1

    flag = 0
    segLine = []
    for l in ls[::-1]:
        flag += l
        segLine += [0 for _ in range(l-1)] + [1]
    segLine = segLine[:-1]
    return segLine

def viterbi(idLine, lm, ds):
    alpha = makeAlpha(idLine, lm , ds)

    # 横に潰す(周辺化)
    alpha = np.sum(alpha, axis=2)

    # viterbi
    alpha = -np.log(alpha)

    bestScore = [None for i in range(len(idLine))]
    bestEdge = [None for i in range(len(idLine))]

    for i in range(len(idLine)):
        scores = []
        for j in range(i+1 if maxL>i else maxL):
            if j==i:
                scores.append(alpha[i][j])
            else:
                scores.append(bestScore[i-(j+1)]+alpha[i][j])
        bestScore[i] = min(scores)
        bestEdge[i] = np.argmin(scores)+1
    
    segLine = []
    flag = len(idLine)
    while len(bestEdge) != 0:
        l = bestEdge[-1]
        segLine += [1]+[0]*(l-1)
        bestEdge = bestEdge[:-l]
        flag -= l
    return segLine[::-1][:-1]

import pickle
class DammyLM:
    def __init__(self):
        self.ngramProbDict = {}
        self.d = pickle.load(open('../model/uniProb.dict','rb'))

    def setNgramProbs(self, inVoc, V, pairs, ds):        
        #for pair in pairs:
        #    self.ngramProbDict[pair] = 0.1
        for pair in pairs:
            p = np.prod([ds.baseProbDict[''.join(ds.ids2chars(word))] for word in pairs[-1]])
            self.ngramProbDict[pair] = p

class DammyDS:
    def __init__(self):
        self.char2id = {'<BOS>':0, '<EOS>':9}

    def getV(self):
        return 10

if __name__ == '__main__':
    import dataset
    import linecache
    lm = DammyLM()
    #ds = dataset.Dataset([linecache.getline('../../data/iphone_train_text.txt', i).strip() 
    #                      for i in range(100)])
    ds = dataset.Dataset([line.strip() for line in open('../../data/iphone_train_text.txt')])

    idLine = ds.idData[4706][1:-1]
    print(ds.ids2chars(idLine))

    setNgram([idLine], lm, ds, None)
    print(viterbi(idLine, lm, ds))

