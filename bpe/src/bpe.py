from collections import defaultdict

limitWordSize = {'train':186345,
                 'test':19581}

def getFreqBigram(data):
    countDict = defaultdict(lambda:0)

    for line in data:
        for i in range(len(line)-1):
            w = line[i]+line[i+1]
            countDict[w] += 1

    k,v = sorted(countDict.items(), key=lambda x:x[1])[-1]
    return k

def updateData(data, x):
    neoData = []
    for line in data:
        neoLine = []
        i = 0
        while i < len(line)-1:
            w = line[i]+line[i+1]
            if w==x:
                w = (line[i][0]+line[i+1][0],)
                neoLine.append(w)
                i+=2
            else:
                neoLine.append(line[i])
                i+=1
        if i<len(line):
            neoLine.append(line[-1])
        neoData.append(neoLine)
    return neoData

d = 'ntcir'
for ty in ['train', 'test']:
    data = [line.strip() for line in open('../../data/%s_%s_text.txt'%(d,ty))]
    data = [[(c,) for c in line] for line in data]

    while True:
        w = getFreqBigram(data)
        data = updateData(data,w)
        wordSize = sum([len(line) for line in data])
        vocSize = len({w for line in data for w in line})
        print('wordSize:', wordSize, w)
        print('vocSize:', vocSize)
        if wordSize<limitWordSize[ty]:
            break

    data = ['　'.join([w[0] for w in line]) for line in data]
    f = open('../data/%s_%s_text_bpe.txt'%(d,ty), 'w')
    for line in data:
        f.write(line+'\n')
    f.close()
