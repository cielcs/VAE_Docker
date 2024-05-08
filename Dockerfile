FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Ubuntuのパッケージリポジトリを日本国内のミラーに変更
RUN sed -i.bak -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list