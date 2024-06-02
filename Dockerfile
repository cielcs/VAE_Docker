FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Ubuntuのパッケージリポジトリを日本国内のミラーに変更
RUN sed -i.bak -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list

# Pythonをインストール
RUN echo "Installing library" &&\
    apt-get -y update && \
    apt-get install -y python3-setuptools\
                       python3.8-dev\
                       python3-pip


RUN cd tensorflow && \
    git checkout v2.8.0 && \
    echo "N" | ./configure
RUN cd tensorflow && bazel build --jobs=8 \
            --config=v2 \
            --copt=-O3 \
            --copt=-m64 \
            --copt=-march=native \
            --config=opt \
            --config=cuda \
            --verbose_failures \
            //tensorflow:tensorflow_cc \
            //tensorflow:install_headers \
            //tensorflow:tensorflow \
            //tensorflow:tensorflow_framework \
            //tensorflow/tools/lib_package:libtensorflow \
            //tensorflow/tools/pip_package:build_pip_package

RUN cd tensorflow && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

RUN pip install /tmp/tensorflow_pkg/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl 
RUN pip install matplotlib
RUN pip install pandas

RUN  mkdir -p /opt/tensorflow/lib && \
     cp -r /tensorflow/bazel-bin/tensorflow/* /opt/tensorflow/lib/ && \
     cd /opt/tensorflow/lib && \
     ln -s libtensorflow_cc.so.2.8.0 libtensorflow_cc.so && \
     ln -s libtensorflow_cc.so.2.8.0 libtensorflow_cc.so.2

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/tensorflow/lib:$LD_LIBRARY_PATH

RUN ldconfig

RUN pip install tensorflow-io

# コンテナ起動時にbashを起動
CMD ["/bin/bash"]