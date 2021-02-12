2020/11 SIGNATE
【第4回_Beginner限定コンペ】自動車の走行距離予測
   https://signate.jp/competitions/355

（期間：1ヶ月, 評価: RMSE=2.729, 順位: 218/465(6), score: 3.4707225, 1位: 2.5375491）

mpg float: 走行距離/ガソリン1ガロン
cylinders varchar: シリンダー
displacement float: 排気量
hotsepower varchar: 馬力
weight float: 重量
acceleration float: 加速度
modelyear varchar: 年式
origin varchar: 起源
carname varchar: 車名

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2回目のSIGNATE投稿。
Questを活用して散布図や箱ひげ図を用いて学習に用いる説明変数の絞り込みを行った。
着々とRMSEのscoreは減少していったが昇格ラインより低いscoreを出すことはできなかった。
恐らく数値のデータは扱えても文字列のデータを学習にうまく使いこなせなかった印象。
jupyterlabで実行した。
初回同様、main関数等用いずに行ったため、コードもあまりきれいでないのがうかがえるが、
あえてそのままにして残しておく。
concatぐらいは使えばよかったなw
近似値問題として扱うか分類問題として扱うかによってもscoreが変わるだろうな...(2021/02/12)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~