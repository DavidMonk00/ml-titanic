import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

data = pd.read_csv("train.csv")
data.head()
sns.catplot('Sex', data=data, kind='count')
sns.lmplot('Age', 'Survived', hue="Pclass", data=data)

features = ["Pclass", "Age"]
label = ["Survived"]

X = data[features]
y = data[label]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.1
)
X_train.head()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, np.ravel(y_train))
predictions = model.predict(X_test)
np.mean(np.ravel(y_test) == predictions)
