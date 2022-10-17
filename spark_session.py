from pyspark import StorageLevel
from pyspark.sql import SparkSession

from log import log


class LocalSparkSession:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.spark = None
        self.sc = None

    def start(self):
        log(f'LocalSparkSession : Starting with {self.num_clusters} clusters')
        self.spark = SparkSession.builder \
            .master(f"local[{self.num_clusters}]") \
            .appName("8INF919D1") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("OFF")
        self.sc = self.spark.sparkContext.getOrCreate()
        rdd1 = self.sc.parallelize([1, 2])
        rdd1.persist(StorageLevel.MEMORY_AND_DISK_2)

    def stop(self):
        log(f'LocalSparkSession : Stopping')
        self.sc.stop()
