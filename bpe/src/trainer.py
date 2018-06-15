import module
import dataset
import lm as L

import numpy as np
from chainer import serializers
from chainer import optimizers
from tqdm import tqdm

import sys

d = sys.argv[1]

batchSize = 64


class Trainer:
    def __init__(self):
        self.ds = dataset.Dataset('../data/%s_train_text_bpe.txt'%d, d=d)
        self.lm = L.LM(d)
        self.opt = optimizers.Adam()
        self.opt.setup(self.lm)

    def train(self, begin=0, end=30):
        for ep in range(begin, end):
            print(ep)
            indices = np.random.permutation(len(self.ds.idData))
            batches = module.pack(indices, batchSize)
            for batch in tqdm(batches):
                self.lm.cleargrads()
                idLines = [self.ds.idData[b] for b in batch]
                loss = self.lm.getLoss(idLines, self.ds.vocab)
                loss.backward()
                self.opt.update()
            self.save(ep)

    def save(self, ep):
        serializers.save_npz('../model/%s_lm_%d.npz'%(d,ep), self.lm)

    def load(self, ep):
        self.lm = L.LM(len(self.ds.char2id)+2, self.ds.id2char)
        serializers.load_npz('../model/%s_lm_%d.npz'%(d,ep) ,self.lm)

if __name__ == '__main__':
    t = Trainer()
    t.train()
