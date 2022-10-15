from log import log
from datetime import datetime
from sklearn.impute import SimpleImputer
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType


class MapReduceIDR3:
    def __init__(self, dataset):
        self.dataset = dataset
        self.labeled_point = None
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.model = None
        self.errors = None
        self.delta_time = None
        log('MapReduceIDR3 : Starting')

    def set_labeled_point(self):
        log(f'MapReduceIDR3 : Setting Labeled Point')
        self.labeled_point = self.dataset.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:]))

    def split(self):
        log(f'MapReduceIDR3 : Splitting')
        (self.training_data, self.test_data) = self.labeled_point.randomSplit([0.5, 0.5])

    def train(self):
        log(f'MapReduceIDR3 : Training')
        time_initial = datetime.now()
        self.model = DecisionTree.trainClassifier(
            self.training_data,
            numClasses=2,
            categoricalFeaturesInfo={},
            impurity='entropy',
            maxDepth=5,
            maxBins=42,
        )
        self.delta_time = datetime.now() - time_initial
        log(f'MapReduceIDR3 : Training time {self.delta_time.total_seconds()} seconds')

    def predict(self):
        log(f'MapReduceIDR3 : Predicting')
        self.predictions = self.model.predict(
            self.test_data.map(lambda x: x.features))
        labels_and_predictions = self.test_data.map(
            lambda lp: lp.label).zip(self.predictions)
        self.errors = labels_and_predictions.filter(
            lambda lp: lp[0] != lp[1]).count() / float(self.test_data.count())
        self.metrics = BinaryClassificationMetrics(labels_and_predictions)
        self.area_under_pr = self.metrics.areaUnderPR
        self.area_under_roc = self.metrics.areaUnderROC

    def get_metrics(self):
        log(f'MapReduceIDR3 : Get metrics')
        return {
            'time': self.delta_time.total_seconds(),
            # 'errors': self.errors,
            # 'area_under_pr': self.area_under_pr,
            # 'area_under_roc': self.area_under_roc,
        }