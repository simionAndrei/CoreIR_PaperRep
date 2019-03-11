from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from skmultilearn.problem_transform import ClassifierChain, LabelPowerset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from util import label_based_accuracy, get_one_hot_from_str_labels

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import json

class FeatureImportanceAnalyzer():

  def __init__(self, sentiment_df, content_df, structural_df, labels, logger):
    self.content_df = content_df
    self.structural_df = structural_df
    self.sentiment_df = sentiment_df
    self.labels = labels
    self.logger = logger
    self.current_df = None
    self.best_model = "45% AdaBoost + 55% RandomForest"

  def _load_models_hyperparams(self):
    with open(self.logger.get_model_file(self.logger.config_dict['BEST_ADA']), 'r') as fp:
      self.adab_hyperparams = json.load(fp)
    with open(self.logger.get_model_file(self.logger.config_dict['BEST_RANDF']), 'r') as fp:
      self.randf_hyperparams = json.load(fp)

  def _split_data(self):

    self.current_df['label'] = self.labels
    one_hot_labels = get_one_hot_from_str_labels(self.current_df.iloc[:, -1].values)
    
    boolean_dict = {True: 1, False: 0}
    self.current_df = self.current_df.replace(boolean_dict)
    X = self.current_df.iloc[:, :-1].values
    y = csr_matrix(one_hot_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
      random_state = 13)
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size = 0.5, 
      random_state = 13)

    self.current_df = self.current_df[0:0]
    del self.current_df

    return X_train, y_train, X_test, y_test


  def _get_model(self, problem_transform = ClassifierChain):

    self._load_models_hyperparams()

    adaboost_model = AdaBoostClassifier(DecisionTreeClassifier())
    adaboost_model.set_params(**self.adab_hyperparams)
    randf_model = RandomForestClassifier()
    randf_model.set_params(**self.randf_hyperparams)

    ensemble_model = problem_transform(
      VotingClassifier(estimators = [('ADA', adaboost_model), ('RANDF', randf_model)], voting='soft', 
      weights = [0.45, 0.55], n_jobs = -1))

    return ensemble_model


  def _evaluate_model(self, y_true, y_pred):

    accuracy = label_based_accuracy(y_true, y_pred)
    self.logger.log("Accuracy label based score {}".format(accuracy))
    self.logger.log("Subset accuracy {}".format(accuracy_score(y_true, y_pred)))
    self.logger.log("Recall {}".format(recall_score(y_true, y_pred, average = 'micro')))
    self.logger.log("Precision {}".format(precision_score(y_true, y_pred, average = 'micro')))
    self.logger.log("F1 {}".format(f1_score(y_true, y_pred, average = 'micro')))


  def _evaluate_feature_set(self):

    X_train, y_train, X_test, y_test = self._split_data()
    model = self._get_model()

    self.logger.log("Start training {} ...".format(self.best_model))
    model.fit(X_train, y_train)
    self.logger.log("Finished training", show_time = True)

    self.logger.log("Start evaluating on test data ...")
    y_pred = model.predict(X_test)
    self._evaluate_model(y_test.toarray(), y_pred.toarray())


  def analyze_sentiment(self):

    self.logger.log("Analyze feature importance by using only sentiment features ...")
    self.current_df = self.sentiment_df.copy(deep = True)
    self._evaluate_feature_set()


  def analyze_content(self):
    
    self.logger.log("Analyze feature importance by using only content features ...")
    self.current_df = self.content_df.copy(deep = True)
    self._evaluate_feature_set()


  def analyze_structural(self):
    
    self.logger.log("Analyze feature importance by using only structural features ...")
    self.current_df = self.structural_df.copy(deep = True)
    self._evaluate_feature_set()


  def analyze_combinations(self):

    self.logger.log("Analyze feature importance by using only content + structural features ...")
    self.current_df = pd.concat([self.content_df, self.structural_df], axis = 1).copy(deep = True)
    self._evaluate_feature_set()

    self.logger.log("Analyze feature importance by using only content + sentiment features ...")
    self.current_df = pd.concat([self.content_df, self.sentiment_df], axis = 1).copy(deep = True)
    self._evaluate_feature_set()

    self.logger.log("Analyze feature importance by using only structural + sentiment features ...")
    self.current_df = pd.concat([self.structural_df, self.sentiment_df], axis = 1).copy(deep = True)
    self._evaluate_feature_set()


  def analyze_individual_importance(self):

    self.logger.log("Analyze individual features importance ...")
    self.current_df = pd.concat([self.structural_df, self.sentiment_df, self.content_df], 
      axis = 1).copy(deep = True)
    feats_name = list(self.current_df.columns.values)
    
    X_train, y_train, X_test, y_test = self._split_data()
    model = self._get_model(problem_transform = LabelPowerset)

    self.logger.log("Start training {} ...".format(self.best_model))
    model.fit(X_train, y_train)
    self.logger.log("Finished training", show_time = True)

    adab_feats_scores = model.classifier.estimators_[0].feature_importances_
    randf_feats_score = model.classifier.estimators_[1].feature_importances_

    results_df = pd.DataFrame(np.stack([adab_feats_scores, randf_feats_score]), 
      columns = feats_name, index = ["AdaBoost", "RandForest"])
    results_df.index.name = "Estimator"

    self.logger.log("Save feature importance scores for each estimator in the ensemble \n {}".format(
      results_df))
    results_df.to_csv(self.logger.get_output_file("feats_imp.csv"))

    self.logger.log("Start evaluating on test data ...")
    y_pred = model.predict(X_test)
    self._evaluate_model(y_test.toarray(), y_pred.toarray())