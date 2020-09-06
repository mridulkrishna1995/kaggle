import pandas as pd
import numpy as np
from titanic.models import random_forest_classifier, decision_tree_classifier
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('E:\\Projects\\kaggle\\data\\titanic\\train.csv')
test_data = pd.read_csv('E:\\Projects\\kaggle\\data\\titanic\\test.csv')

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = train_data[features]
X_test = test_data[features]
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X = imp.transform(X)
X_test = imp.transform(X_test)

predictions = random_forest_classifier(X, y, X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('E:\\Projects\\kaggle\\output\\titanic\\random_forest_4.csv', index=False)