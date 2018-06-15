from collections import defaultdict
import pickle

bos, eos = ('<BOS>','<EOS>')

class Dataset:
    def __init__(self, trainDataPath, testDataPath, trainLabelPath, testLabelPath, d):
        # dataは半角スペースでtokenizeされているもの
        self.vocab = None
        self.char2id, self.id2char = pickle.load(open('../../uws/model/%s/ids.dict'%d,'rb'))
        self.idData_train = self.getIdData(trainDataPath)
        self.idData_test = self.getIdData(testDataPath)
        self.label_train = self.getLabel(trainLabelPath)
        self.label_test = self.getLabel(testLabelPath)

    def getLabel(self, path):
        data = [int(line.strip()) for line in open(path) if line.strip()]
        return data

    def getIdData(self, path):
        data = [line.strip() for line in open(path) if line.strip()]

        idData = []

        bos_id = self.char2id[bos]
        eos_id = self.char2id[eos]

        for line in data:
            ws = line.split('　')
            idLine = [tuple([self.char2id[c] for c in w]) for w in ws if w]
            
            idLine = [(bos_id,)] + idLine + [(eos_id,)]
            idData.append(idLine)

        return idData

if __name__ == '__main__':
    ds = Dataset('../data/iphone_train_bpe_text.txt')
    #ds = Dataset('../../data/iphone_test_text.txt')
    print(ds.idData[:10])
    print(ds.vocab[:10])
