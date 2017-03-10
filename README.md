# face_prediction

## 学習(ファインチューニングなし)
学習方法は以下の通り、-g はグラボある人

```
~$ python facePredictionTraining.py -p ./images/ -g 0
```
画像を入れたフォルダの階層は以下のようにする。　　
各個人の写真の入ったフォルダの上のフォルダまでを-pで指定する。  
  
./images/  
　　├ 0_the_others/*  
　　├ akimoto/*  
　　├ hashimoto/*  
　　├ ikuta/*  
　　├ ikoma/*  
　　├ nishino/*  
　　└ shiraishi/*  

## 学習(ファインチューニングあり)  
別途alexnetのcaffeモデルをダウンロードして、それをpklファイルに変換します。

```
~$ python
> #読み込むcaffeモデルとpklファイルを保存するパス
> loadpath = "bvlc_alexnet.caffemodel"
> savepath = "./chainermodels/alexnet.pkl"
 
> from chainer.links.caffe import CaffeFunction
> alexnet = CaffeFunction(loadpath)
 
> import _pickle as pickle
> pickle.dump(alexnet, open(savepath, 'wb'))
```
学習する時は、保存したpklファイルを指定してあげるだけです。
```
~$ python facePredictionTraining.py -p ./images/ -g 0 -cmp alexnet.pkl
```

## 予測時　　
学習したモデルファイルを使って、予測する時は以下のようにします。

```
~$ python facePrediction.py -m my_output_7.model -p 画像のパス
```  

予測時は、opencvの顔認識でどこに顔があるかを判別し、そこを切り取り、学習済みのモデルファイルに投げて誰かを判別するようになっています。
なので画像は顔の切り取りをしておく必要はありません。
現時点ではファインチューニングした際のモデルに関してそのまま予測することは出来ないと思いますが、モデルの定義の部分を少し変えればできるようになると思います。
