from log import log
from datetime import datetime
from sklearn.impute import SimpleImputer
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType


class CrossValidationPySpark:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None
        self.labeled_point = None
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.model = None
        self.errors = None
        self.delta_time = None
        log('MapReduceIDR3 : Starting')

    def train(self):
        # https://gist.github.com/colbyford/7758088502211daa90dbc1b51c408762
        time_initial = datetime.now()
        dt = DecisionTreeClassifier()
        pipeline = Pipeline(stages=[dt, ])
        param_grid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [4, 5, 6]).addGrid(dt.maxBins, [42]).build()
            # .addGrid(dt.minInstancesPerNode, [1, 2, 3]) \
            # .addGrid(dt.maxBins, [16, 32, 64]) \
            # .build()
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=param_grid,
                                  evaluator=BinaryClassificationEvaluator(),
                                  numFolds=2)
        self.model = crossval.fit(self.dataset.df)
        self.delta_time = datetime.now() - time_initial

    def get_metrics(self):
        return {
            'time': self.delta_time.total_seconds(),
            'errors': self.errors,
            'area_under_pr': self.area_under_pr,
            'area_under_roc': self.area_under_roc,
        }