FROM ubuntu:20.04

COPY . /classification_distribuee

WORKDIR ./classification_distribuee

RUN apt update
RUN apt -y upgrade
RUN apt -y full-upgrade

RUN apt install curl mlocate wget default-jdk -y

RUN wget https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz

RUN tar xvf spark-3.3.0-bin-hadoop3.tgz
RUN mv spark-3.3.0-bin-hadoop3/ /opt/spark

RUN echo "export SPARK_HOME=/opt/spark" > ~/.bashrc
RUN echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" > ~/.bashrc

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9
RUN apt-get install -y python3-pip python3-dev

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools==57.4.0
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install jupyter

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0",  "--no-browser"]

# docker build -t "docker push marcelovasconcellos/pyspark-distributed-classification:latest:latest" .
# docker tag pyspark-distributed-classification:latest marcelovasconcellos/pyspark-distributed-classification:latest
# docker push marcelovasconcellos/pyspark-distributed-classification:latest
# docker run -d -p 8888:8888 -m 1GB --cpus=4 "pyspark:Dockerfile"
