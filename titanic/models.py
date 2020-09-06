from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def random_forest_classifier(X, y, X_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    return predictions

def decision_tree_classifier(X, y, X_test):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(X, y)
    predictions = model.predict(X_test)
    return predictions
