from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from skmultilearn.problem_transform import ClassifierChain

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

    print(default_estimator.get_params().keys())

    random_grid_search = RandomizedSearchCV(estimator = default_estimator, 
      param_distributions = hyperparams_grid, n_iter = 100, cv = 3, verbose=2, 
      random_state=13, n_jobs = -1)

    random_grid_search.fit(X_valid, y_valid)
    logger.log("Hyperparameter random grid-search done", show_time = True, tabs = 1)

    best_params = random_grid_search.best_params_
    params_filename = type(base_estimator).__name__ + "_params_" + logger.get_time_prefix()
    params_filename += ".json"

    old_keys = list(best_params.keys())
    for key in old_keys:
      best_params[key.replace("classifier__", "")] = best_params.pop(key)

    logger.log("Best params are {}".format(best_params), tabs = 1)

    with open(logger.get_model_file(params_filename), 'w') as fp:
      json.dump(best_params, fp)

    base_estimator.set_params(**best_params)

  logger.log("Start training  {} ...".format(type(base_estimator).__name__))
  classifier.fit(X_train, y_train)
  logger.log("Finish training model", show_time = True)

  logger.log("Evaluating model on test data ...")
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


def test_ada_boost(X_train, y_train, X_test, y_test, logger, 
  X_valid = None, y_valid = None):

  default_dtc = DecisionTreeClassifier()
  base_estimator = AdaBoostClassifier(base_estimator = default_dtc)

  n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)]
  learning_rate = [0.01, 0.05, 0.1, 0.3, 1]
  base_estimator__criterion = ["gini", "entropy"]
  base_estimator__splitter = ["best", "random"]
  base_estimator__max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  base_estimator__max_depth.append(None)
  base_estimator__min_samples_split = [2, 5, 10]
  base_estimator__min_samples_leaf = [1, 2, 4]
  base_estimator__max_features = ['sqrt', 'log2', None]

  hyperparams_grid = {'classifier__n_estimators' : n_estimators,
                      'classifier__learning_rate' : learning_rate,
                      'classifier__base_estimator__criterion' : base_estimator__criterion,
                      'classifier__base_estimator__splitter' : base_estimator__splitter,
                      'classifier__base_estimator__max_depth' : base_estimator__max_depth,
                      'classifier__base_estimator__min_samples_split' : base_estimator__min_samples_split,
                      'classifier__base_estimator__min_samples_leaf' : base_estimator__min_samples_leaf,
                      'classifier__base_estimator__max_features' : base_estimator__max_features}

  classifier = test_model(X_train, y_train, X_test, y_test, logger, base_estimator, 
    hyperparams_grid, X_valid, y_valid)

  return classifier


def test_combiner_svm_ada(X_train, y_train, X_test, y_test, logger):

  with open(logger.get_model_file(logger.config_dict['BEST_SVM']), 'r') as fp:
    svm_hyperparams = json.load(fp)
  with open(logger.get_model_file(logger.config_dict['BEST_ADA']), 'r') as fp:
    adaboost_hyperparams = json.load(fp)

  svm_model = SVC(verbose = True, probability = True)
  svm_model.set_params(**svm_hyperparams)
  adaboost_model = AdaBoostClassifier(DecisionTreeClassifier())
  adaboost_model.set_params(**adaboost_hyperparams)

  ensemble_weights = logger.config_dict['ENS1_WEIGHTS']
  ensemble_model = ClassifierChain(
    VotingClassifier(estimators = [('SVM', svm_model), ('ADA', adaboost_model)], voting='soft', 
    weights = ensemble_weights, n_jobs = -1))

  logger.log("Start training average ensemble {:.2f} SVM + {:.2f} AdaBoost ...".format(
    ensemble_weights[0], ensemble_weights[1]))
  final_ensemble.fit(X_train, y_train)
  logger.log("Finish training average ensemble", show_time = True)

  logger.log("Evaluating model on test data ...")
  y_pred = final_ensemble.predict(X_test)

  accuracy = label_based_accuracy(y_test.toarray(), y_pred.toarray())
  logger.log("Accuracy label based score {}".format(accuracy))
  logger.log("Subset accuracy {}".format(accuracy_score(y_test.toarray(), y_pred.toarray())))
  logger.log("Recall {}".format(recall_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))
  logger.log("Precision {}".format(precision_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))
  logger.log("F1 {}".format(f1_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))

  return final_ensemble


def test_combiner_svm_randf(X_train, y_train, X_test, y_test, logger):

  with open(logger.get_model_file(logger.config_dict['BEST_SVM']), 'r') as fp:
    svm_hyperparams = json.load(fp)
  with open(logger.get_model_file(logger.config_dict['BEST_RANDF']), 'r') as fp:
    randf_hyperparams = json.load(fp)

  svm_model = SVC(verbose = True, probability = True)
  svm_model.set_params(**svm_hyperparams)
  randf_model = RandomForestClassifier()
  randf_model.set_params(**randf_hyperparams)

  ensemble_weights = logger.config_dict['ENS2_WEIGHTS']
  ensemble_model = ClassifierChain(
    VotingClassifier(estimators = [('SVM', svm_model), ('RANDF', adaboost_model)], voting='soft', 
    weights = ensemble_weights, n_jobs = -1))

  logger.log("Start training average ensemble {:.2f} SVM + {:.2f} RandForest ...".format(
    ensemble_weights[0], ensemble_weights[1]))
  final_ensemble.fit(X_train, y_train)
  logger.log("Finish training average ensemble", show_time = True)

  logger.log("Evaluating model on test data ...")
  y_pred = final_ensemble.predict(X_test)

  accuracy = label_based_accuracy(y_test.toarray(), y_pred.toarray())
  logger.log("Accuracy label based score {}".format(accuracy))
  logger.log("Subset accuracy {}".format(accuracy_score(y_test.toarray(), y_pred.toarray())))
  logger.log("Recall {}".format(recall_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))
  logger.log("Precision {}".format(precision_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))
  logger.log("F1 {}".format(f1_score(y_test.toarray(), y_pred.toarray(), average = 'micro')))

  return final_ensemble