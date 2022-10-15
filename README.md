# 8INF919 Devoir1 - Classification distribuée par arbre de decision


8INF919_Devoir1_Classification-distribuee-par-arbre-de-decision

```
docker build -t "pyspark-distributed-classification:latest" .
```

```
docker run -d -p 8888:8888 --memory=8GB "pyspark-distributed-classification:latest"
```

```
docker tag pyspark-distributed-classification:latest marcelovasconcellos/pyspark-distributed-classification:latest
```

```
docker push marcelovasconcellos/pyspark-distributed-classification:latest
```