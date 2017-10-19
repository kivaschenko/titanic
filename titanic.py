#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kivaschenko
"""

# import libraries
import numpy as np 
import pandas as pd 
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


DATA_DIR = "dev/titanic"

train_file = "{0}/train.csv".format(DATA_DIR)
test_file = "{0}/test.csv".format(DATA_DIR)
submission_file = "{0}/gender_submission.csv".format(DATA_DIR)

df = pd.concat([pd.read_csv(train_file), pd.read_csv(test_file)])
print(df.info())




#   To fill missed values of main features:
df["Embarked"].fillna("S", inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df.loc[ df['Fare'] <= 7.91, 'Fare']                               = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
df.loc[ df['Fare'] > 31, 'Fare']                                  = 3
df['Fare'] = df['Fare'].astype(int)
    
df.loc[df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[df['Age'] > 64, 'Age'] = 4

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

#  Assign cabin availibility
df['has_cabin'] = df.loc[:,"Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#  Get length of names: name_length
df['name_length'] = df['Name'].apply(len)

#   To count the family size: 
df['family_size'] = df.loc[:,'SibSp'] + df.loc[:,'Parch'] + 1
df['is_alone'] = 0
df.loc[df['family_size'] == 1, 'is_alone'] = 1

for name_column in ['Sex', 'Embarked', 'title']:
    for key, value in zip(df[name_column].unique(), 
        list(range(len(df[name_column].unique())))):
        df.loc[df[name_column] == key, name_column] = value
    df[name_column] = df[name_column].astype(float)


df = df.drop([  'Age',
                'Fare', 
                'Cabin', 
                'Ticket',
                'Name',
                'name_length',
                'SibSp',
                'PassengerId'], axis=1, inplace=True)

print('df past dropping: {}'.format(df.info()))

train = df[(df.Survived == 1) | (df.Survived == 0)]
survived = train['Survived'][:].values
print("The target set -- survived.shape:{}".format(survived.shape))

train = train.drop('Survived', axis=1)
print("The new train set:{}".format(train.head()))

test = df[(df.Survived != 1) & (df.Survived != 0)]
test.drop('Survived', axis=1, inplace=True)

X = train.values
y = survived.ravel()
X_pred = test.values


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_pred_scaled = scaler.transform(X_pred)

print("Shape X_scaled: {0}. Shape y: {1}. Shape X_pred_scaled : {2}"\
      .format(X_scaled.shape, y.shape, X_pred_scaled.shape))
print("X_scaled[:10]: {0}.\n\n\n y[:10]:{1}.\n\n\n X_pred_scaled[:10]:{2}" \
    .format(X_scaled[:10], y[:10], X_pred_scaled[:10]))

param_grid = {'n_estimators': [n for n in range(20, 200, 20)],
              'max_depth'   : [d for d in range(3,7)],
              'max_features': [f for f in range(2, 7)]}
print("Parameter grid:\n{}".format(param_grid))
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, 
                                                n_jobs=-1), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=21)
grid_search.fit(X_train, y_train)
print("Test set score: {:.5f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))

forest = grid_search.best_estimator_

# forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=7, max_features=4, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
#             oob_score=False, random_state=42, verbose=0, warm_start=False)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.5f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(forest.score(X_test, y_test)))

kfold = KFold(n_splits=5)
scores = cross_val_score(forest, X, y, cv=kfold)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.5f}".format(scores.mean()))


################ To write the results in file ####################
submission = pd.read_csv(submission_file)
submission.iloc[:, 1] = forest.predict(X_pred_scaled)
submission.to_csv('random_forest_clf_titanic_subm.csv', index=False)

