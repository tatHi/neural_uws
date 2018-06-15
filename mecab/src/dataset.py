from collections import defaultdict
import pickle

bos, eos = ('<BOS>','<EOS>')

class Dataset:
    def __init__(self, dataPath, data=None, d=None):
        # dataは半角スペースでtokenizeされているもの
        if data is None:
            data = [line.strip() for line in open(dataPath) if line.strip()]

        self.vocab = None
        self.char2id, self.id2char = pickle.load(open('../../uws/model/%s/ids.dict'%d,'rb'))
        self.idData = []

        self.setIdData(data)

    def setIdData(self, data):
        bos_id = self.char2id[bos]
        eos_id = self.char2id[eos]

        self.vocab = set()

        for line in data:
            ws = line.split('　')
            idLine = [tuple([self.char2id[c] for c in w]) for w in ws if w]
            
            idLine = [(bos_id,)] + idLine + [(eos_id,)]
            self.idData.append(idLine)

            self.vocab |= set(idLine)
        self.vocab = list(self.vocab)

if __name__ == '__main__':
    ds = Dataset('../data/iphone_train_bpe_text.txt')
    #ds = Dataset('../../data/iphone_test_text.txt')
    print(ds.idData[:10])
    print(ds.vocab[:10])
