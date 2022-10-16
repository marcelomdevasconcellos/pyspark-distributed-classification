import sys
import environ
import datetime
import pandas as pd
from spark_session import LocalSparkSession
from dataset import Dataset
from mr_id3 import MapReduceIDR3
from pyspark.mllib.tree import DecisionTree

from log import log
from sklearn.impute import SimpleImputer
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler



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
MULTIPLICATION_FACTORS = env('NUMBER_OF_CORES', default='1,10,20,30,40,50,60,70,80,90,100')

numbers_of_cores = [int(n) for n in NUMBER_OF_CORES.split(',')]
number_of_core = max(numbers_of_cores)

multiplication_factors = [int(n) for n in MULTIPLICATION_FACTORS.split(',')]
multiplication_factor = max(multiplication_factors)

metrics = []

filename = f'dataset/adult_{multiplication_factor}x.data'

for number_of_cores in numbers_of_cores:

    spark = LocalSparkSession(number_of_cores)
    spark.start()

    dataset = Dataset(spark.spark, filename, num_fields, categorical_fields, target)
    dataset.load()
    dataset.select_only_numerical_features()

    df = dataset.df

    mr_id3 = MapReduceIDR3(df)
    mr_id3.train()

    metric = mr_id3.get_metrics()
    metric['dataset_rows'] = df.count()
    metric['dataset_size_num'] = multiplication_factor
    metric['dataset_size'] = sys.getsizeof(dataset.df_pandas)
    metric['number_of_cores'] = number_of_cores
    metrics.append(metric)
    log(f"Metrics: Clusters {metric['number_of_cores']} - Dataset size {metric['dataset_size_num']}x - Time {metric['time']} seconds")

    spark.stop()

now = str(datetime.datetime.now()).replace(':', '_').replace(',', '_').replace('.', '_').replace(' ', '_')
df = pd.DataFrame.from_dict(metrics)
df.to_csv(f'results/{ENVIRONMENT}_DATASIZE_{multiplication_factor}_{now}.csv')





