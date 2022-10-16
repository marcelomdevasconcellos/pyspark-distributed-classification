#!/bin/bash

apt update
apt -y upgrade
apt -y full-upgrade

apt install curl mlocate wget default-jdk -y
apt install git -y

wget https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz

tar xvf spark-3.3.0-bin-hadoop3.tgz
mv spark-3.3.0-bin-hadoop3/ /opt/spark

echo "export SPARK_HOME=/opt/spark" > ~/.bashrc
echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" > ~/.bashrc

apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install -y python3.9
apt-get install -y python3-pip python3-dev

python3 -m pip install --upgrade pip
python3 -m pip install setuptools==57.4.0
python3 -m pip install --no-cache-dir -r requirements.txt
