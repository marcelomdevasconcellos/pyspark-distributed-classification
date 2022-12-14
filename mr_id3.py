from datetime import datetime

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

from log import log


class MapReduceIDR3:
    def __init__(self, df):
        self.df = df
        self.labeled_point = None
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.model = None
        self.errors = None
        self.train_time = None
        log('MapReduceIDR3 : Starting')

    def __set_labeled_point(self):
        log(f'MapReduceIDR3 : Setting Labeled Point')
        self.labeled_point = self.df.rdd.map(lambda line: LabeledPoint(line[0], line[1:]))

    def __split(self):
        log(f'MapReduceIDR3 : Splitting')
        (self.training_data, self.test_data) = self.labeled_point.randomSplit([0.5, 0.5])

    def train(self):
        self.__set_labeled_point()
        self.__split()
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
        self.train_time = datetime.now() - time_initial
        log(f'MapReduceIDR3 : Training time {self.train_time.total_seconds()} seconds')

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
            'time': self.train_time.total_seconds(),
            # 'errors': self.errors,
            # 'area_under_pr': self.area_under_pr,
            # 'area_under_roc': self.area_under_roc,
        }
