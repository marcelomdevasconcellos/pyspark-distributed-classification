import os

import pandas as pd
from spark_session import LocalSparkSession
from dataset import Dataset
from mr_id3 import MapReduceIDR3

if not os.path.exists('results'):
    os.makedirs('results')

num_fields = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', ]

categorical_fields = [
    'workclass', 'education',
    'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'native_country', ]

target = 'label'
filename = '/Users/marcelovasconcellos/PycharmProjects/8INF919_Devoir1_Classification-distribuee-par-arbre-de-decision/dataset/adult.data'

clusters = [4, ]  # list(range(1, 2))
multiply = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # list(range(1, 2))
metrics = []

if __name__ == '__main__':

    for n_clusters in clusters:
        for m_factor in multiply:
            spark = LocalSparkSession(n_clusters)
            spark.start()

            dataset = Dataset(spark.spark, filename, num_fields, categorical_fields, target)
            dataset.load()
            dataset.one_hot_encode_categorical_fields()
            dataset.multiply_dataset(m_factor)

            mr_id3 = MapReduceIDR3(dataset)
            mr_id3.set_labeled_point()
            mr_id3.split()
            mr_id3.train()
            mr_id3.predict()

            metric = mr_id3.get_metrics()
            metric['length_rows'] = dataset.df.count()
            metric['m_factor'] = m_factor
            metric['n_clusters'] = n_clusters
            metrics.append(metric)

            spark.stop()

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv('results/metrics.csv')


