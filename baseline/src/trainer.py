import dataset
import module
import numpy as np
import chainer
from chainer import optimizers
from chainer import serializers
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys

data = sys.argv[1]
mode = sys.argv[2]
classifier = sys.argv[3]

if classifier=='dan':
    import dan
    cl = dan
elif classifier=='bilstm':
    import bilstm
    cl = bilstm
elif classifier=='lstm':
    import lstm
    cl = lstm

batchSize = 64

# 学習用分割済みテキスト
trainDataPath = '../data/%s_train_text_%s.txt'%(data,mode)

# 評価用分割済みテキスト
testDataPath = '../data/%s_test_text_%s.txt'%(data,mode)

# 学習用ラベル
trainLabelPath = '../../data/%s_train_label.txt'%data

# 評価用ラベル
testLabelPath = '../../data/%s_test_label.txt'%data

# 単語分散表現辞書
wordDictPath = None if mode=='base' else '../data/%s_wordVec_%s.dict'%(data,mode)

class Trainer:
    def __init__(self):
        self.ds = dataset.Dataset(trainDataPath,testDataPath,trainLabelPath,testLabelPath, data)
        self.model = cl.Classifier()
        self.model.setDict(wordDictPath)

        self.opt = optimizers.Adam()
        self.opt.setup(self.model)

    def train(self, begin=0, end=100):
        print('loss\tmicro\tmacro')
        for ep in range(begin, end):
            print(ep)
            indices = np.random.permutation(len(self.ds.idData_train))
            batches = module.pack(indices, batchSize)
            accLoss = 0
            for batch in tqdm(batches):
                self.model.cleargrads()
                idLines = [self.ds.idData_train[b] for b in batch]
                ts = [self.ds.label_train[b] for b in batch]
                
                loss = self.model.forward(idLines, ts)
                loss.backward()
                self.opt.update()

                accLoss += loss.data
            accLoss /= len(batches)
            #self.save(ep)
            micro,macro = self.evaluate()
            print('%f\t%f\t%f'%(accLoss,micro,macro))
            self.writeResult(ep,accLoss,micro,macro)

    def save(self, ep):
        serializers.save_npz('../model/%s_%s_%s_%d.npz'%(data, classifier, mode, ep), self.model)

    def load(self, ep):
        serializers.load_npz('../model/%s_%s_%s_%d.npz'%(data, classifier, mode, ep), self.model)

    def evaluate(self, ep=None):
        if ep is not None:
            self.load(ep)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                zs = np.argmax(self.model.forward(self.ds.idData_test).data, axis=1)
        ts = self.ds.label_test
        f1_micro = f1_score(ts, zs, average='micro')
        f1_macro = f1_score(ts, zs, average='macro')

        return f1_micro, f1_macro

    def writeResult(self, ep, loss, micro, macro):
        f = open('../model/result_%s_%s_%s.txt'%(data, classifier,mode), 'a')
        if ep==0:
            f.write('ep\tloss\tmicro\tmacro\n')
        f.write('%d\t%f\t%f\t%f\n'%(ep,loss,micro,macro))

if __name__ == '__main__':
    t = Trainer()
    t.train()
