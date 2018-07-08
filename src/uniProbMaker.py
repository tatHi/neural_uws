import pickle
from collections import defaultdict
import math
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tp','--textPath',type=str,help='text path to segmentate')
parser.add_argument('-ml','--maxLength',default=8,
                    type=int,help='max length of word')
parser.add_argument('-rp','--resultPath',type=str,help='path to save')
args = parser.parse_args()

maxL = args.maxLength

data = [line.strip() for line in open(args.textPath)]

uniDict = defaultdict(lambda:0)    # 1文字の辞書

for line in data:
    for i in range(len(line)):
        u = line[i]
        uniDict[u] += 1

# count
uniAll = sum([uniDict[u] for u in uniDict])

uniProbDict = {c:uniDict[c]/uniAll for line in data for c in line}

writePath = args.resultPath+'/uniProb.dict'
pickle.dump(uniProbDict, open(writePath, 'wb'))

print('done:',writePath)
