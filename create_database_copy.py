from dataset import Dataset
from spark_session import LocalSparkSession

number_of_cores = 4
filename = 'dataset/adult.data'
multiplication_factors = [1000, ]

target = 'label'
num_fields = [
    'age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
    'hours_per_week', ]

categorical_fields = [
    'workclass', 'education',
    'marital_status', 'occupation', 'relationship',
    'race', 'sex', 'native_country', ]

spark = LocalSparkSession(number_of_cores)
spark.start()

dataset = Dataset(spark.spark, filename, num_fields, categorical_fields, target)
for f in multiplication_factors:
    dataset.create_copy(f'dataset/adult_{f}x.data', f, update_filename=False)
    print(f'dataset/adult_{f}x.data OK!')

spark.stop()











