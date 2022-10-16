from log import log
from datetime import datetime
from sklearn.model_selection import cross_validate
from sklearn import decomposition, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DecisionTreeSklearn:
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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        log('DecisionTreeSklearn : Starting')

    def __set_x_y(self):
        log(f'DecisionTreeSklearn : Setting X and y')
        self.X = self.df.drop(columns=['label', ])
        self.y = self.df['label']

    def __split(self):
        log(f'DecisionTreeSklearn : Splitting')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.5, random_state=42)

    def crossvalidation_train(self, parameters=False):
        log(f'DecisionTreeSklearn : Cross Validation Training')
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
        self.model.fit(self.X_train, self.y_train)
        self.delta_time = datetime.now() - time_initial
        log(f'DecisionTreeSklearn : Cross Validation Training time {self.delta_time.total_seconds()} seconds')

    def train(self, parameters={'criterion': 'entropy', }):
        log(f'DecisionTreeSklearn : Training')
        self.__set_x_y()
        self.__split()
        time_initial = datetime.now()
        clf = DecisionTreeClassifier(**parameters)
        self.model = clf.fit(self.X_train, self.y_train)
        self.delta_time = datetime.now() - time_initial
        log(f'DecisionTreeSklearn : Training time {self.delta_time.total_seconds()} seconds')

    def get_metrics(self):
        log(f'DecisionTreeSklearn : Getting metrics')
        return {
            'time': self.delta_time.total_seconds(),
            # 'best_estimator': self.model.best_estimator_,
            # 'best_score': self.model.best_score_,
            # 'best_params': self.model.best_params_,
        }