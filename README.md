## Dockerの習得も兼ねてTensorflowを用いたVAEの生成システム
おそらく使用する環境やライブラリ  
Docker, Python3, NVIDIA/CUDA, tensorflow,  

### Docker環境作り方
1. Dockerfileに必要な要素を記述する~ベースイメージの指定やライブラリのインストール。#まだ分かってないから後で書こう
2. Dockerfileを元にDocker imageをビルドする。
   - コマンド "docker build -t <イメージ名>:<タグ> ."
3. Dockerコンテナの作成。
   - docker run --gpus all -it --rm -v [ホストディレクトリの絶対パス]:[コンテナの絶対パス] [イメージ名] [コマンド]
4. コンテナを毎回破棄するか使い続けるか
   - 使い続けるなら
   - docker start <コンテナ名またはコンテナID> #停止していたコンテナを起動
   - docker exec -it my_container /bin/bash #コンテナに入る
   - 今回は毎回破棄する予定。
#### コマンド一覧
- docker images #作成したDockerイメージの一覧を表示
- docker ps -a #作成したDockerコンテナ一覧を表示
- docker build -t my_cuda_image:1.0 . # imageビルドの例
- docker run --gpus all -it --rm -v D:\Programing\vaesystem:/home vae_image:1.0 /bin/bash #コンテナ作成の例。消去オプションとかマウントするパスが書いてある。
- docker container prune #未使用コンテナを消去
- docker rmi [イメージID] # docker image消去

#### Dokerfileの書き方
- とりあえずFROM ~でベースイメージ指定
- RUNでコマンド実行
- TODO 追記する

