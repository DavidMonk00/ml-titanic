import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class ModelTester:
    def __init__(self):
        pass

    def loadData(self, filename):
        self.data = pd.read_csv(filename)

    def cleanData(self, func=lambda data: data):
        self.data = func(self.data)

    def listFeatures(self):
        return self.data.columns

    def setFeatures(self, features):
        self.features = features

    def setLabel(self, label):
        self.label = label

    def setModel(self, model):
        self.model = model

    def plotCorrelation(self):
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(self.data.astype(float).corr(), linewidths=0.1, vmax=1.0,
                    square=True, cmap=colormap, linecolor='white', annot=True)
        plt.show()

    def fit(self):
        X = self.data[self.features]
        y = self.data[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)
        self.model.fit(self.X_train, np.ravel(self.y_train))

    def evaluate(self, test_data=None):
        if test_data is None:
            predictions = self.model.predict(self.X_test)
            return np.mean(np.ravel(self.y_test) == predictions)
        else:
            predictions = self.model.predict(test_data[self.features])
            output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
            output.to_csv('my_submission.csv', index=False)

    def testModel(self, model, params):
        self.setModel(
            model
        )
        self.model.set_params(**params)
        self.fit()
        return self.evaluate()

    def crossValidate(self, param_grid):
        def plotting(X, Y, Error, model):
            plt.figure(figsize=[12, 5])
            plt.title("ROC_AUC score:" + model)
            plt.xlabel('param_value')
            plt.ylabel('ROC_AUC score')
            plt.plot(X, Y, 'bo-', color='b', ms=5, label="ROC_AUC score")
            plt.fill_between(X, Y - 1.96*Error, Y + 1.96*Error, facecolor='g', alpha=0.6, label="95% confidence interval")
            plt.legend()
            plt.grid()
            plt.show()

        searcher = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
        searcher.fit(self.X_train, np.ravel(self.y_train))
        plotting(
            param_grid[0][list(param_grid[0].keys())[0]],
            searcher.cv_results_['mean_test_score'],
            searcher.cv_results_['std_test_score'],
            "KNN"
        )

        print("Best parameters:{}".format(searcher.best_params_))
        print("Best parameters:{}".format(searcher.best_score_))

        plt.show()


features = [
    "Pclass", "Sex_bool", "CategoricalAge", "CategoricalFare", "Parch",
    "SibSp", "embarked_C", "embarked_Q", "embarked_S", "Family_size",
    "in_cabin", "Title", "Alone"
]


def preProcess(data):
    data["Sex_bool"] = data["Sex"] == "male"
    data = data.drop(columns=["Sex"])
    data["Family_size"] = data["SibSp"] + data["Parch"] + 1
    data["Alone"] = data["Family_size"] == 1
    data["in_cabin"] = data["Cabin"].isna() is False
    data = data.drop(columns=["Cabin"])
    data = data.drop(columns=["Ticket"])
    embarked = pd.get_dummies(data["Embarked"], prefix="embarked")
    data = data.join(embarked)
    data = data.drop(columns=["Embarked"])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['CategoricalFare'] = pd.qcut(data['Fare'], 8, labels=False)
    data['Age'] = data['Age'].fillna(data["Age"].mean())
    data['Age'] = data['Age'].astype(int)
    data['CategoricalAge'] = pd.qcut(data['Age'], 4, labels=False)

    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    data['Title'] = data['Name'].apply(get_title)
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                           'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    data = data.drop(columns=["Name"])
    return data


def main():
    test_data = pd.read_csv("test.csv")
    mt = ModelTester()
    mt.loadData("train.csv")
    mt.cleanData(preProcess)
    mt.setFeatures(
        features
    )
    mt.setLabel(["Survived"])
    print(mt.listFeatures())
    # mt.plotCorrelation()

    print(mt.testModel(
        KNeighborsClassifier(),
        {
            'n_jobs': -1,
            'n_neighbors': 11
        }
    ))
    param_grid = [{'n_neighbors': list(np.arange(1, 100, 3))}]
    # mt.crossValidate(param_grid)

    print(mt.testModel(
        RandomForestClassifier(),
        {
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 1
        }
    ))
    param_grid = [{'max_depth': list(np.arange(1, 20, 1))}]
    # mt.crossValidate(param_grid)

    print(mt.testModel(
        GradientBoostingClassifier(),
        {
            'n_estimators': 100,
        }
    ))

    param_grid = [{'n_estimators': list(np.arange(1, 200, 3))}]
    # mt.crossValidate(param_grid)
    mt.evaluate(test_data=preProcess(test_data))


if __name__ == '__main__':
    main()
