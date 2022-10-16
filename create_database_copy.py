import environ
from dataset import Dataset
from spark_session import LocalSparkSession

env = environ.Env()
environ.Env.read_env()

NUMBER_OF_CORES = env('NUMBER_OF_CORES', default='1,2,3,4')
MULTIPLICATION_FACTORS = env('NUMBER_OF_CORES', default='1,10,20,30,40,50,60,70,80,90,100')

numbers_of_cores = [int(n) for n in NUMBER_OF_CORES.split(',')]
multiplication_factors = [int(n) for n in MULTIPLICATION_FACTORS.split(',')]

filename = 'dataset/adult.data'

metrics = []

target = 'label'
num_fields = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', ]

categorical_fields = [
    'workclass', 'education',
    'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'native_country', ]

spark = LocalSparkSession(max(numbers_of_cores))
spark.start()

dataset = Dataset(spark.spark, filename, num_fields, categorical_fields, target)
for f in multiplication_factors:
    dataset.create_copy(f'dataset/adult_{f}x.data', f, update_filename=False)
    print(f'dataset/adult_{f}x.data OK!')

spark.stop()











