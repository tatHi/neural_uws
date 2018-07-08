def read(path):
    data = [line.strip() for line in open(path, 'r')]
    idsData = []
    textData = []

    for line in data:
        line = line.split(':')
        ids = (line.pop(0), line.pop(0))
        if line:
            idsData.append(ids)
            textData.append(':'.join(line))

    return idsData, textData

def pack(arr, size):
    batch = [arr[i:i+size] for i in range(0, len(arr), size)]
    return batch

def segmentIdLine(idLine, segLine):
    segIdLine = []
    tmp = []
    for i,s in zip(idLine, segLine):
        tmp.append(i)
        if s == 1:
            segIdLine.append(tuple(tmp))
            tmp = []
    tmp.append(idLine[-1])
    segIdLine.append(tuple(tmp))

    return segIdLine

def showSegSentence(line, segLine):
    ws = []
    tmp = ''
    for c,s in zip(line, segLine):
        tmp += c
        if s==1:
            ws.append(tmp)
            tmp = ''
    tmp += line[-1]
    ws.append(tmp)
    print('_'.join(ws))


