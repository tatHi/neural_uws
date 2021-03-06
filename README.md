# neural_uws
Unsupervised Word Segmentation with Neural Language Model  
ニューラル言語モデルを用いた教師なし単語分割  

[発表原稿](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=190355&item_no=1&page_id=13&block_id=8)
(一部の人にしかダウンロードできないらしいです．)   
発表資料: slide.pdf(第236回NL研)    
不明点などあればメールか何かで聞いてください．

特許をとっているNPYLMを元にした研究です．  
念のため研究目的以外での利用は控えてください．  
https://twitter.com/daiti_m/status/851810748263157760  

まだGPU対応してないです．  

## 環境
python 3.6.4  

```
$ pip install chainer==4.1.0  
$ pip install numpy==1.13.3  
$ pip install gensim==3.4.0
```

## 使い方
### 作業場所の作成
```
$ mkdir result
$ cd src
```

  
### 文字分散表現の事前学習
```
$ python charVecTrainer.py --textPath ../data/kokoro.txt \
                           --resultPath ../result \
                           --embedSize 30 \
                           --windowSize 3 \
                           --epoch 100 \
```
  
### 離散ユニグラム辞書の作成
```
$ python uniProbMaker.py --textPath ../data/kokoro.txt \
                         --resultPath ../result \
                         --maxLength 8
```

  
### 分割学習
```
$ python segmentater.py --mode train \
                        --textPath ../data/kokoro.txt \
                        --pretrainPath ../result \
                        --resultPath ../result \
                        --beginEpoch 0 \
                        --endEpoch 50\
                        --batchSize 32 \
                        --samplingSizeK 100 \
                        --showSeg
```

  
### 前向きアルゴリズムによる分割
学習データで未知の文字が含まれなければ，どのようなテキストでも可．  
```
python segmentater.py --mode seg \
                      --textPath ../data/kokoro.txt \
                      --pretrainPath ../result \
                      --resultPath ../result \
                      --batchSize 8 \
                      > ../result/segedData.txt
```

  
### 単語分割に対して分散表現を計算して辞書化
```
python segmentater.py --mode vecAssign \
                      --pretrainPath ../result \
                      --resultPath ../result \
                      --segedTextPath ../result/segedData.txt
```
