FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Ubuntuのパッケージリポジトリを日本国内のミラーに変更
RUN sed -i.bak -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list

# Pythonをインストール
RUN echo "Installing library" &&\
    apt-get -y update && \
    apt-get install -y python3-setuptools\
                       python3.8-dev\
                       python3-pip

RUN pip install pandas matplotlib

# コンテナ起動時にbashを起動
CMD ["/bin/bash"]