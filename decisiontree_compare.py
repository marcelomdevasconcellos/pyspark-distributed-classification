import datetime
import sys

import environ
import pandas as pd

from dataset import Dataset
from decisiontree_pyspark import DecisionTreePySpark
from decisiontree_sklearn import DecisionTreeSklearn
from spark_session import LocalSparkSession

num_fields = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', ]

categorical_fields = [
    'workclass', 'education',
    'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'native_country', ]

target = 'label'

env = environ.Env()
environ.Env.read_env()

ENVIRONMENT = env('ENVIRONMENT', default='LOCAL')
NUMBER_OF_CORES = env('NUMBER_OF_CORES', default='1,2,3,4')
MULTIPLICATION_FACTORS = env('MULTIPLICATION_FACTORS', default='1,10,20,30,40,50,60,70,80,90,100')

numbers_of_cores = [int(n) for n in NUMBER_OF_CORES.split(',')]
number_of_core = max(numbers_of_cores)

multiplication_factors = [int(n) for n in MULTIPLICATION_FACTORS.split(',')]
multiplication_factor = max(multiplication_factors)

spark = LocalSparkSession(number_of_core)
spark.start()

metrics = []

for f in multiplication_factors:
    dataset = Dataset(
        spark.spark,
        f'dataset/adult.data',
        num_fields, categorical_fields, target)
    dataset.create_copy(f'dataset/adult_{f}x.data', f, update_filename=True)
    dataset.load()
    dataset.select_only_numerical_features()

    df = dataset.df
    df_pandas = dataset.df_pandas

    metric_dict = {'dataset_size_num': f, 'dataset_size': sys.getsizeof(df_pandas)}

    # PySpark
    dt_pyspark = DecisionTreePySpark(df)
    dt_pyspark.train()
    dt_pyspark.predict()
    m = dt_pyspark.get_metrics()
    metric_dict['pyspark_time'] = m['time']
    metric_dict['pyspark_train_time'] = m['train_time']
    metric_dict['pyspark_predict_time'] = m['predict_time']
    dt_pyspark = None

    # SKLearn
    if df_pandas != None:
        dt_sklearn = DecisionTreeSklearn(df_pandas)
        dt_sklearn.train()
        dt_sklearn.predict()
        m = dt_sklearn.get_metrics()
        metric_dict['sklearn_time'] = m['time']
        metric_dict['sklearn_train_time'] = m['train_time']
        metric_dict['sklearn_predict_time'] = m['predict_time']
    else:
        metric_dict['sklearn_time'] = -1
        metric_dict['sklearn_train_time'] = -1
        metric_dict['sklearn_predict_time'] = -1

    dt_sklearn = None

    metrics.append(metric_dict)

    now = str(datetime.datetime.now()).replace(':', '_').replace(',', '_').replace('.', '_').replace(' ', '_')
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(f'results/{ENVIRONMENT}_COMPARE_{number_of_core}_{f}_{now}_TEMP.csv')

    dataset.delete_copy(f'dataset/adult_{f}x.data')

spark.stop()

now = str(datetime.datetime.now()).replace(':', '_').replace(',', '_').replace('.', '_').replace(' ', '_')
df = pd.DataFrame.from_dict(metrics)
df.to_csv(f'results/{ENVIRONMENT}_COMPARE_{number_of_core}_{now}.csv')
