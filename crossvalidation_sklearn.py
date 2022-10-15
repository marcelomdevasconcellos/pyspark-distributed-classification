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
    def __init__(self, df):
        self.df = df
        self.training_data = None
        self.test_data = None
        self.predictions = None
        self.model = None
        self.errors = None
        self.delta_time = None
        self.X = None
        self.y = None
        log('CrossValidationSkLearn : Starting')

    def set_x_y(self):
        log(f'CrossValidationSkLearn : Setting X and y')
        self.X = self.df.drop(columns=['label', ])
        self.y = self.df['label']

    def train(self, parameters=False):
        log(f'CrossValidationSkLearn : Training')
        dt = DecisionTreeClassifier(criterion='entropy')
        pipe = Pipeline(
            steps=[('dt', dt), ])
        if not parameters:
            parameters = dict(
                #df__criterion=['entropy', ],
                dt__max_depth=[10, 20, 30, 40, 50, 60, 70],
                #dt__min_samples_split=[1, 2, 3],
                #dt__max_features=[16, 32, 64]
            )
        time_initial = datetime.now()
        self.model = GridSearchCV(pipe, parameters)
        self.model.fit(self.X, self.y)
        self.delta_time = datetime.now() - time_initial
        log(f'CrossValidationSkLearn : Training time {self.delta_time.total_seconds()} seconds')

    def get_metrics(self):
        log(f'CrossValidationSkLearn : Getting metrics')
        return {
            'time': self.delta_time.total_seconds(),
            # 'best_estimator': self.model.best_estimator_,
            # 'best_score': self.model.best_score_,
            # 'best_params': self.model.best_params_,
        }