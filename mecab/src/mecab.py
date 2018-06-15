import MeCab
import sys

d = sys.argv[1]

tagger = MeCab.Tagger('-Owakati')

for ty in ['train', 'test']:
    data = [tagger.parse(line.strip()).strip() for line in open('../../data/%s_%s_text.txt'%(d,ty))]

    wordSize = sum([len(line.split(' ')) for line in data])
    print(wordSize)

    f = open('../data/%s_%s_text_mecab.txt'%(d,ty), 'w')
    for line in data:
        f.write(line.replace(' ','　')+'\n')
    f.close()
