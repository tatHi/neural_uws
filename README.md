# neural_uws
Unsupervised Word Segmentation with Neural Language Model
ニューラル言語モデルを用いた教師なし単語分割

7/9を目処に整理します．  
[発表原稿](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=190355&item_no=1&page_id=13&block_id=8)  
発表資料:gitに後ほどcommit  

特許をとっているNPYLMを元にした研究です．  
念のため研究目的以外での利用は控えてください．


  
作業場所の作成
```
$ mkdir result
$ cd src
```

  
文字分散表現の事前学習
```
$ python charVecTrainer.py --textPath ../data/kokoro.txt \
                           --resultPath ../result \
                           --embedSize 30 \
                           --windowSize 3 \
                           --epoch 100 \
                           --batchSize 64
```
  
離散ユニグラム辞書の作成
```
$ python uniProbMaker.py --textPath ../data/kokoro.txt \
                         --resultPath ../result \
                         --maxLength 8
```

  
分割学習
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

  
前向きアルゴリズムによる分割
```
python segmentater.py --mode seg \
                      --textPath ../data/kokoro.txt \
                      --pretrainPath ../result \
                      --resultPath ../result \
                      --batchSize 8 \
                      > ../result/segedData.txt
```
