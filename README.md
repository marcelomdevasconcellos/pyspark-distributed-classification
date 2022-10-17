# 8INF919 Devoir1 - Classification distribuée par arbre de decision


8INF919_Devoir1_Classification-distribuee-par-arbre-de-decision

```
docker build -t "pyspark-distributed-classification:latest" .
```

```
docker run -d -p 8888:8888 --memory=8GB "pyspark-distributed-classification:latest"
```

```
python3 create_database_copy.py
```
```
python3 garantie-du-passage-a-l-echelle.py
```
```
python3 necessite-de-la-distribution-de-l-apprentissage.py
```
```
python3 decisiontree_compare.py
```
```
python3 garantie-du-passage-a-l-echelle.py && python3 necessite-de-la-distribution-de-l-apprentissage.py && python3 decisiontree_compare.py
```

```
docker tag pyspark-distributed-classification:latest marcelovasconcellos/pyspark-distributed-classification:latest
```

```
docker push marcelovasconcellos/pyspark-distributed-classification:latest
```