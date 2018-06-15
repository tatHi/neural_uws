import pickle

root = '../model/ntcir/'

for ep in range(0,75):
    ds = pickle.load(open(root+'ds_%d.pickle'%ep,'rb'))
    ds.getLambda()
