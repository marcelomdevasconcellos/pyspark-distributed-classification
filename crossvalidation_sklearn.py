from log import log
from datetime import datetime
from sklearn.model_selection import cross_validate
from sklearn import decomposition, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class CrossValidationSkLearn:
    def __init__(self, dataset):
        self.dataset = dataset
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.model = None
        self.errors = None
        self.delta_time = None
        self.X = None
        self.y = None
        log('MapReduceIDR3 : Starting')

    def split(self):
        self.training_data, self.test_data = self.labeled_point.randomSplit([0.7, 0.3])

    def set_x_y(self):
        self.X = self.dataset.df.toPandas().drop(columns=['label', ])
        self.y = self.dataset.df.toPandas()['label']

    def train(self):
        # https://ai.plainenglish.io/hyperparameter-tuning-of-decision-tree-classifier-using-gridsearchcv-2a6ebcaffeda?gi=eee67beedcab
        time_initial = datetime.now()
        dt = DecisionTreeClassifier()
        pipe = Pipeline(
            steps=[('dt', dt), ])
        parameters = dict(dt__max_depth=[4, 5, 6],
                          #dt__min_samples_split=[1, 2, 3],
                          #dt__max_features=[16, 32, 64]
                          )
        self.model = GridSearchCV(pipe, parameters)
        self.model.fit(self.X, self.y)
        self.delta_time = datetime.now() - time_initial

    def get_metrics(self):
        return {
            'time': self.delta_time.total_seconds(),
            'best_estimator': self.model.best_estimator_,
            'best_score': self.model.best_score_,
            'best_params': self.model.best_params_,
        }