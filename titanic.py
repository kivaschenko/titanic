#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kivaschenko
"""

# import libraries
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


data_dir = "dev/titanic"

train_file = "{0}/train.csv".format(data_dir)
test_file = "{0}/test.csv".format(data_dir)
submission_file = "{0}/gender_submission.csv".format(data_dir)

df = pd.concat([pd.read_csv(train_file), pd.read_csv(test_file)])
print(df.info())

#   To fill missed values of main features:
df["Embarked"].fillna("S", inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

#   To count the family size: 
df['family_size'] = df['SibSp'] + df['Parch'] + 1

# Define function to extract titles from passenger names
def get_title(name_pass):
    title_search = re.search(' ([A-Za-z]+)\.', name_pass)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

df['title'] = df['Name'].apply(get_title)


# Group all non-common titles into one single grouping "Rare"
df['title'] = df['title'].replace(['Lady', 
                                    'Countess',
                                    'Capt', 
                                    'Col',
                                    'Don', 
                                    'Dr', 
                                    'Major', 
                                    'Rev', 
                                    'Sir', 
                                    'Jonkheer', 
                                    'Dona'],  'Rare')
#   Change the rare  title meet at often titles:
df['title'] = df['title'].replace('Mlle', 'Miss')
df['title'] = df['title'].replace('Ms', 'Miss')
df['title'] = df['title'].replace('Mme', 'Mrs')

for key, value in zip(df.title.unique(), list(range(len(df.title.unique())))):
    df.loc[df["title"] == key, "title"] = value

#  Get length of names: name_length
df['name_length'] = df['Name'].apply(len)


# Get some clusters of the age and fare of passengers: age_fare_clusters
age_fare_clusters = np.array(df.loc[:,['Age', 'Fare']])
kmeans = KMeans()
kmeans.fit(age_fare_clusters)
df['age_fare_clusters'] = kmeans.labels_


# The binary features add:
def get_dummie_columns(df, name_column):
    """Convert to binar number of value categories current column"""
    
    df_dummie = pd.get_dummies(df[name_column][:], prefix=name_column)
    df_dummie = pd.concat([df[:],df_dummie.iloc[:,:-1]], axis=1)
    
    return df_dummie

name_column = ['Sex','Embarked']

#   The bypassing the needing column name list
for col in name_column:
    x = df.loc[:,:]
    df = get_dummie_columns(x, col)
    df.drop(col, axis=1, inplace=True)


df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

train = df[(df.Survived == 1) | (df.Survived == 0)]
survived = train['Survived'][:].values
print("The target set -- survived.shape:{}".format(survived.shape))

train = train.drop(['PassengerId', 'Survived'], axis=1)
print("The new train set:{}".format(train.head()))

test = df[(df.Survived != 1) & (df.Survived != 0)]
test.drop(['PassengerId', 'Survived'], axis=1, inplace=True)

X = train.values
y = survived
X_pred = test.values
print("Shape X: {0}. Shape y: {1}. Shape X_pred : {2}"\
      .format(X.shape, y.shape, X_pred.shape))

param_grid = {'n_estimators': [n for n in range(10, 110, 10)],
              'max_features': [f for f in range(2, 12)]}
print("Parameter grid:\n{}".format(param_grid))
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)
grid_search.fit(X_train, y_train)
print("Test set score: {:.5f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))

forest = grid_search.best_estimator_
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.5f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(forest.score(X_test, y_test)))

kfold = KFold(n_splits=5)
scores = cross_val_score(forest, X, y, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.5f}".format(scores.mean()))


################ To write the results in file ####################
submission = pd.read_csv(submission_file)
submission.iloc[:, 1] = forest.predict(X_pred)
submission.to_csv('random_forest_clf_titanic_subm.csv', index=None)

# Shape X: (891, 12). Shape y: (891,). Shape X_pred : (418, 12)
# Parameter grid:
# {'max_features': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
# Test set score: 0.83857
# Best parameters: {'max_features': 5, 'n_estimators': 30}
# Best cross-validation score: 0.84132
# Best estimator:
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features=5, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=-1,
#             oob_score=False, random_state=42, verbose=0, warm_start=False)
# Accuracy on training set: 0.98503
# Accuracy on test set: 0.83857
# Cross-validation scores: [ 0.7877095   0.80898876  0.81460674  0.79213483  0.84831461]
# Average cross-validation score: 0.81035
