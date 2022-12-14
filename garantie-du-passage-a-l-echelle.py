import sys
import environ
import datetime
import pandas as pd

from dataset import Dataset
from decisiontree_pyspark import DecisionTreePySpark
from log import log
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

metrics = []

spark = LocalSparkSession(number_of_core)
spark.start()

for f in multiplication_factors:
    dataset = Dataset(
        spark.spark,
        f'dataset/adult.data',
        num_fields, categorical_fields, target)
    dataset.create_copy(f'dataset/adult_{f}x.data', f, update_filename=True)
    dataset.load()
    dataset.select_only_numerical_features()

    df = dataset.df

    mr_id3 = DecisionTreePySpark(df)
    mr_id3.train()

    metric = mr_id3.get_metrics()
    metric['dataset_rows'] = df.count()
    metric['dataset_size_num'] = f
    metric['dataset_size'] = sys.getsizeof(dataset.df_pandas)
    metric['number_of_cores'] = number_of_core
    metrics.append(metric)

    log(f"Metrics: Clusters {metric['number_of_cores']} - Dataset size {metric['dataset_size_num']}x - Time {metric['time']} seconds")

    # now = str(datetime.datetime.now()).replace(':', '_').replace(',', '_').replace('.', '_').replace(' ', '_')
    # df = pd.DataFrame.from_dict(metrics)
    # df.to_csv(f'results/Garantie-du-passage-a-l-echelle_{ENVIRONMENT}_{number_of_core}_CORES_{f}X_{now}_TEMP.csv')

    dataset.delete_copy(f'dataset/adult_{f}x.data')

spark.stop()

now = str(datetime.datetime.now()).replace(':', '_').replace(',', '_').replace('.', '_').replace(' ', '_')
df = pd.DataFrame.from_dict(metrics)
df.to_csv(f'results/Garantie-du-passage-a-l-echelle_{ENVIRONMENT}_{number_of_core}_CORES_{now}.csv')
