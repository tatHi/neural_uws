# neural_uws
Unsupervised Word Segmentation with Neural Language Model
ニューラル言語モデルを用いた教師なし単語分割

7/9を目処に整理します．  
[発表原稿](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=190355&item_no=1&page_id=13&block_id=8)  
発表資料:gitに後ほどcommit  

特許をとっているNPYLMを元にした研究です．  
念のため研究目的以外での利用は控えてください．

```
$ mkdir result
$ cd src
```

```
$ python charVecTrainer.py --textPath ../data/kokoro.txt --resultPath ../result --embedSize 30 --windowSize 3 --epoch 100 --batchSize 64
```

```
$ python uniProbMaker.py --textPath ../data/kokoro.txt  --resultPath ../result --maxLength 8
```
