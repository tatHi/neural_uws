import numpy as np 
from tqdm import tqdm
import dataset
import pickle
import module
from chainer import optimizers
from time import time

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

args = parser.parse_args()

# set dataset 
ds = dataset.Dataset(args.textPath)
pickle.dump((ds.char2id, ds.id2char), open(args.resultPath+'/ids.dict','wb'))
print('id-dict saved')

bos = ['<BOS>']
eos = ['<EOS>']
data4w2v = [bos+list(line)+eos for line in ds.data]

from gensim.models import word2vec
model = word2vec.Word2Vec(data4w2v, size=args.embedSize, min_count=0, window=3, iter=args.epoch)
print('train done')

charVecTable = np.stack([model.wv[ds.id2char[i]] for i in range(len(ds.id2char))])
np.save(args.resultPath+'/charEmbed.npy', charVecTable)

print('model saved')
