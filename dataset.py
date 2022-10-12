import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import seaborn as sns
sns.set()
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


class Dataset:
    def __init__(self, spark, filename, num_fields, categorical_fields, target):
        self.spark = spark
        self.filename = filename
        self.num_fields = num_fields
        self.categorical_fields = categorical_fields
        self.target = target
        self.schema = None
        self.df = None
        self.df_assembler = None
        log('Dataset : Starting')

    def load(self):
        log(f'Dataset : Loading Dataset {self.filename}')
        self.schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", StringType(), True),
            StructField("education", StringType(), True),
            StructField("education_num", StringType(), True),
            StructField("marital_status", StringType(), True),
            StructField("occupation", StringType(), True),
            StructField("relationship", StringType(), True),
            StructField("race", StringType(), True),
            StructField("sex", StringType(), True),
            StructField("capital_gain", StringType(), True),
            StructField("capital_loss", StringType(), True),
            StructField("hours_per_week", StringType(), True),
            StructField("native_country", StringType(), True),
            StructField("label", StringType(), True), ])
        self.df = self.spark.read.csv(
            self.filename,
            header=False,
            schema=self.schema
        )
        self.df.createOrReplaceTempView("Adults")
        self.df = self.spark.sql("""
            SELECT  CASE
                    WHEN TRIM(label) = '>50K' THEN 1
                    ELSE 0 END AS label,
                    INT(age) AS age, 
                    TRIM(workclass) AS workclass,
                    INT(fnlwgt) AS fnlwgt,
                    TRIM(education) AS education,
                    INT(education_num) AS education_num,
                    TRIM(marital_status) AS marital_status,
                    TRIM(occupation) AS occupation,
                    TRIM(relationship) AS relationship,
                    TRIM(race) AS race,
                    TRIM(sex) AS sex,
                    INT(capital_gain) AS capital_gain,
                    INT(capital_loss) AS capital_loss,
                    INT(hours_per_week) AS hours_per_week,
                    TRIM(native_country) AS native_country
            FROM Adults""")

    def one_hot_encode(self, column_name):
        distinct_values = self.df.select(column_name) \
            .distinct().rdd \
            .flatMap(lambda x: x).collect()
        for distinct_value in distinct_values:
            function = udf(lambda item:
                           1 if item == distinct_value else 0,
                           IntegerType())
            new_column_name = column_name + '_' + distinct_value.replace('-', '_').replace('?', 'unknown').lower()
            self.df = self.df.withColumn(new_column_name, function(col(column_name)))
        return self.df.drop(column_name)

    def one_hot_encode_categorical_fields(self):
        for column_name in self.categorical_fields:
            self.df = self.one_hot_encode(column_name)

    def string_indexer(self):
        from pyspark.ml.feature import StringIndexer
        from pyspark.ml.feature import VectorAssembler
        columns = self.num_fields
        for column_name in self.categorical_fields:
            columns.append(column_name+'_idx')
            indexer = StringIndexer(inputCol=column_name, outputCol=column_name+'_idx')
            indexer_model = indexer.fit(self.df)
            self.df = indexer_model.transform(self.df)
        assembler = VectorAssembler(inputCols=columns, outputCol='features')
        self.df_assembler = assembler.transform(self.df)
        self.df_assembler = self.df_assembler.select('features', 'label')
        self.df = self.df.select(columns + ['label'])

    def multiply_dataset(self, multiply):
        appended = self.df
        for _ in list(range(multiply)):
            appended = appended.union(appended)
        self.df = appended