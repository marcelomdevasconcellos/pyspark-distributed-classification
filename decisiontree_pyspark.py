from datetime import datetime

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

from log import log

DEFAULT_PARAMETERS = {
    'numClasses': 2,
    'categoricalFeaturesInfo': {},
    'impurity': 'entropy',
    'maxDepth': 5,
    'maxBins': 42,
}


class DecisionTreePySpark:
    def __init__(self, df):
        self.df = df
        self.labeled_point = None
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.model = None
        self.errors = None
        self.train_time = 0
        self.predict_time = 0
        self.df_assembler = None
        log(f'DecisionTreePySpark : Starting')

    def __set_labeled_point(self):
        log(f'DecisionTreePySpark : Setting Labeled Point')
        self.labeled_point = self.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:]))

    def __split(self):
        log(f'DecisionTreePySpark : Splitting')
        (self.training_data, self.test_data) = self.labeled_point.randomSplit([0.5, 0.5])

    def __assemble_features(self):
        log(f'DecisionTreePySpark : Assembling')
        # self.df = self.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:])).toDF()
        # self.map = self.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:]))
        columns = [col for col in self.df.columns if col != 'label']
        assembler = VectorAssembler(inputCols=columns, outputCol='features')
        self.df_assembler = assembler.transform(self.df)

    def crossvalidation_train(self, parameters=False):
        log(f'DecisionTreePySpark : Cross Validation Training')
        self.__set_labeled_point()
        self.__split()
        self.__assemble_features()
        dt = DecisionTree()
        if not parameters:
            parameters = ParamGridBuilder() \
                .addGrid(dt.trainClassifier.maxDepth, [10, 20, 30, 40, 50, 60, 70]).build()
        time_initial = datetime.now()
        crossval = CrossValidator(estimator=DecisionTree.trainClassifier(),
                                  estimatorParamMaps=parameters,
                                  evaluator=BinaryClassificationEvaluator(),
                                  numFolds=2)
        self.model = crossval.fit(self.training_data)
        self.train_time = datetime.now() - time_initial
        self.train_time = self.train_time.total_seconds()
        log(f'DecisionTreePySpark : Cross Validation Training time {self.train_time} seconds')

    def train(self, parameters=DEFAULT_PARAMETERS):
        log(f'DecisionTreePySpark : Training')
        self.__set_labeled_point()
        self.__split()
        self.__assemble_features()
        time_initial = datetime.now()
        self.model = DecisionTree.trainClassifier(
            self.training_data,
            **parameters
        )
        self.train_time = datetime.now() - time_initial
        self.train_time = self.train_time.total_seconds()
        log(f'DecisionTreePySpark : Train time {self.train_time} seconds')

    def predict(self):
        log(f'DecisionTreePySpark : Predicting')
        time_initial = datetime.now()
        self.predictions = self.model.predict(
            self.test_data.map(lambda x: x.features))
        self.predict_time = datetime.now() - time_initial
        self.predict_time = self.predict_time.total_seconds()
        log(f'DecisionTreePySpark : Predict time {self.predict_time} seconds')
        # labels_and_predictions = self.test_data.map(
        #     lambda lp: lp.label).zip(self.predictions)
        # self.errors = labels_and_predictions.filter(
        #     lambda lp: lp[0] != lp[1]).count() / float(self.test_data.count())
        # self.metrics = BinaryClassificationMetrics(labels_and_predictions)
        # self.area_under_pr = self.metrics.areaUnderPR
        # self.area_under_roc = self.metrics.areaUnderROC

    def get_metrics(self):
        log(f'DecisionTreePySpark : Get metrics')
        return {
            'train_time': self.train_time,
            'predict_time': self.predict_time,
            'time': self.train_time + self.predict_time,
        }
