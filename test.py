from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from util import label_based_accuracy

import numpy as np

import json


def test_model(X_train, y_train, X_test, y_test, logger, base_estimator, hyperparams_grid, 
  X_valid = None, y_valid = None):

  assert (X_valid is None and y_valid is None) or (X_valid is not None and y_valid is not None)

  if X_valid is not None:

    logger.log("Start hyperparameter random grid-search for {}".format(type(base_estimator).__name__))
    default_estimator = ClassifierChain(base_estimator)

    random_grid_search = RandomizedSearchCV(estimator = default_estimator, 
      param_distributions = hyperparams_grid, n_iter = 100, cv = 3, verbose=2, 
      random_state=13, n_jobs = -1)

    random_grid_search.fit(X_valid, y_valid)
    logger.log("Hyperparameter random grid-search done", show_time = True, tabs = 1)

    best_params = random_grid_search.best_params_
    params_filename = type(base_estimator).__name__ + "_params_" + logger.get_time_prefix()
    params_filename += ".json"

    logger.log("Best params are {}".format(best_params), tabs = 1)

    with open(logger.get_model_file(params_filename), 'w') as fp:
      json.dump(best_params, fp)

    old_keys = list(best_params.keys())
    for key in old_keys:
      best_params[key.replace("classifier__", "")] = best_params.pop(key)

    base_estimator.set_params(**best_params)

  classifier = ClassifierChain(base_estimator)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)

  accuracy = label_based_accuracy(y_test.toarray(), y_pred.toarray())
  logger.log("Accuracy label based score {}".format(accuracy))
  logger.log("Subset accuracy {}".format(accuracy_score(y_test.toarray(), y_pred.toarray())))
  logger.log("Recall {}".format(recall_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))
  logger.log("Precision {}".format(precision_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))
  logger.log("F1 {}".format(f1_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))

  return classifier


def test_svm(X_train, y_train, X_test, y_test, logger, 
  X_valid = None, y_valid = None):

  base_estimator = SVC()

  C = [10**i for i in range(4)]
  kernel = ['linear', 'rbf']
  gamma = [0.001, 0.0001, 'auto', 'scale']
  hyperparams_grid = {'classifier__C': C,
                      'classifier__kernel': kernel,
                      'classifier__gamma': gamma}

  classifier = test_model(X_train, y_train, X_test, y_test, logger, base_estimator, 
    hyperparams_grid, X_valid, y_valid)

  return classifier


def test_random_forest(X_train, y_train, X_test, y_test, logger, 
  X_valid = None, y_valid = None):

  base_estimator = RandomForestClassifier()

  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  max_features = ['auto', 'sqrt']
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  min_samples_split = [2, 5, 10]
  min_samples_leaf = [1, 2, 4]
  bootstrap = [True, False]
  hyperparams_grid = {'classifier__n_estimators': n_estimators,
                      'classifier__max_features': max_features,
                      'classifier__max_depth': max_depth,
                      'classifier__min_samples_split': min_samples_split,
                      'classifier__min_samples_leaf': min_samples_leaf,
                      'classifier__bootstrap': bootstrap}

  classifier = test_model(X_train, y_train, X_test, y_test, logger, base_estimator, 
    hyperparams_grid, X_valid, y_valid)

  return classifier


'''
[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 22.3min finished
[2019.03.07-14:58:49] Best params are {'classifier__n_estimators': 1200, 'classifier__min_samples_split': 5, 'classifier__min_samples_leaf': 2, 'classifier__max_features': 'auto', 'classifier__max_depth': 40, 'classifier__bootstrap': False}
'''