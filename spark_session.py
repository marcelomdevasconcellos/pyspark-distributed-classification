import matplotlib
import numpy as np
import pandas as pd
from log import log
from pyspark import StorageLevel
import pyspark
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


class LocalSparkSession:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.spark = None
        self.sc = None

    def start(self):
        self.spark = SparkSession.builder \
            .config("spark.sql.debug.maxToStringFields", "-1") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .master(f"local[{self.num_clusters}]") \
            .appName("8INF919D1") \
            .getOrCreate()
        self.sc = self.spark.sparkContext.getOrCreate()
        rdd1 = self.sc.parallelize([1, 2])
        rdd1.persist(StorageLevel.MEMORY_AND_DISK_2)

    def stop(self):
        self.sc.stop()
