# 8INF919 Devoir1 - Classification distribuée par arbre de decision

8INF919 – APPRENTISSAGE AUTOMATIQUE POUR LE BIGDATA - HIVER 2022

Professeur A. Bouzouane

UNIVERSITÉ DU QUÉBEC À CHICOUTIMI

Département d’Informatique et de Mathématique

MEDM28048006 - Marcelo Medeiros de Vasconcellos

## 1. Pour construire une image docker 

vous devez avoir installé docker desktop sur votre ordinateur

```
docker build -t "pyspark-distributed-classification:latest" .
```

## 2. Pour exécuter l'image docker

```
docker run -p 8888:8888 --memory=8GB "pyspark-distributed-classification:latest"
```

## 3. Exécution des commandes

Vous devrez avoir accès à Docker-CLI pour exécuter les commandes directement dans le terminal.

```
python3 garantie-du-passage-a-l-echelle.py
```
```
python3 necessite-de-la-distribution-de-l-apprentissage.py
```
```
python3 decisiontree_compare.py
```

Pour plus d'informations, lisez le fichier : 8INF919-Devoir1-MEDM28048006-Marcelo-Medeiros-de-Vasconcellos.pdf