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

    def assemble_features(self):
        # self.df = self.dataset.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:])).toDF()
        # self.map = self.dataset.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:]))
        columns = [col for col in self.dataset.df.columns if col != 'label']
        assembler = VectorAssembler(inputCols=columns, outputCol='features')
        df_assembler = assembler.transform(self.df)

    def train(self):
        # https://gist.github.com/colbyford/7758088502211daa90dbc1b51c408762
        time_initial = datetime.now()
        columns = [col for col in self.dataset.df.columns if col != 'label']
        assembler = VectorAssembler(inputCols=columns, outputCol='features')
        df_assembler = assembler.transform(self.dataset.df)
        dt = DecisionTreeClassifier()
        # pipeline = Pipeline(stages=[dt, ])
        param_grid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [10, 20, 30, 40, 50, 60, 70]).build()
            #.addGrid(dt.maxBins, [32, ]).build()
            # .addGrid(dt.minInstancesPerNode, [1, 2, 3]) \
            # .addGrid(dt.maxBins, [16, 32, 64]) \
            # .build()
        crossval = CrossValidator(estimator=DecisionTreeClassifier(),
                                  estimatorParamMaps=param_grid,
                                  evaluator=BinaryClassificationEvaluator(),
                                  numFolds=2)
        self.model = crossval.fit(df_assembler)
        self.delta_time = datetime.now() - time_initial

    def get_metrics(self):
        return {
            'time': self.delta_time.total_seconds(),
            # 'errors': self.errors,
            # 'area_under_pr': self.area_under_pr,
            # 'area_under_roc': self.area_under_roc,
        }