#!/bin/bash

sudo apt update
sudo apt -y upgrade
sudo apt -y full-upgrade

sudo apt install curl mlocate wget default-jdk -y
sudo apt install git -y

sudo snap install --classic code

sudo wget https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz

sudo tar xvf spark-3.3.0-bin-hadoop3.tgz
sudo mv spark-3.3.0-bin-hadoop3/ /opt/spark

sudo echo "export SPARK_HOME=/opt/spark" > ~/.bashrc
sudo echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" > ~/.bashrc

sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.9
sudo apt-get install -y python3-pip python3-dev
sudo apt install python3.8-venv
sudo python3 -m pip install --upgrade pip
python3 -m venv venv
source venv/bin/activate
pip install setuptools
pip install wheel
pip install --no-cache-dir -r requirements.txt

sudo snap install jupyter

cp env_cloud .env
