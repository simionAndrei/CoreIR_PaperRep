from features import StructuralFeatures, SentimentFeatures, ContentFeatures
from data_preproc import DataPreprocessor
from logger import Logger

from util import hamming_score

import pandas as pd
import numpy as np
import itertools

from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.sparse import csr_matrix

def _compute_feats(msdialog_dict, logger):
  structural_feats_extractor = StructuralFeatures(logger)
  structural_df = structural_feats_extractor.compute_features(msdialog_dict)
  structural_df.to_csv(logger.get_data_file("structural.csv"), index = False)
  logger.log("Structural Features: \n {}".format(structural_df.head()))

  sentiment_feats_extractor = SentimentFeatures(logger)
  sentiment_df = sentiment_feats_extractor.compute_features(msdialog_dict)
  sentiment_df.to_csv(logger.get_data_file("sentiment.csv"), index = False)
  logger.log("Sentiment Features: \n {}".format(sentiment_df.head()))

  content_feats_extractor = ContentFeatures(logger)
  content_df = content_feats_extractor.compute_features(msdialog_dict)
  content_df.to_csv(logger.get_data_file("content.csv"), index = False)
  logger.log("Content Features: \n {}".format(content_df.head()))

  return structural_df, sentiment_df, content_df


def _read_feats(logger):
  logger.log("Read structural features ...")
  structural_df = pd.read_csv(logger.get_data_file("structural.csv"))
  logger.log("Read sentiment features ...")
  sentiment_df  = pd.read_csv(logger.get_data_file("sentiment.csv"))
  logger.log("Read content features ...")
  content_df    = pd.read_csv(logger.get_data_file("content.csv"))
  logger.log("Read all features ...")

  return structural_df, sentiment_df, content_df



if __name__ == '__main__':
  logger = Logger(show = True, html_output = True, config_file = "config.txt")

  data_preprocessor = DataPreprocessor(logger.config_dict['DATA_FILE'], logger)
  msdialog_dict = data_preprocessor.get_preprocess_data(0.9)

  if logger.config_dict['COMPUTE_FEATS']:
    structural_df, sentiment_df, content_df = _compute_feats(msdialog_dict, logger)
  else:
    structural_df, sentiment_df, content_df = _read_feats(logger)
        
  feats_df = pd.concat([structural_df, sentiment_df, content_df], axis = 1)
  feats_df['label'] = data_preprocessor.final_tags
  feats_df.to_csv(logger.get_data_file("dialog_dataset.csv"), index = False)

  selected_data = feats_df[feats_df['label'].isin(data_preprocessor.selected_tags)]


  ### WORK IN PROGRESS BELLOW 

  string_labels = selected_data.iloc[:, -1].values
  atomic_labels_list = [str_label.split() for str_label in string_labels]
  atomic_labels_list = list(itertools.chain.from_iterable(atomic_labels_list))
  atomic_labels_list = set(atomic_labels_list)

  one_hot_labels = []
  for string_label in string_labels:
    one_hot_labels.append([1 if tag in string_label else 0 for tag in atomic_labels_list])


  X = selected_data.iloc[:, :-1].astype(float).values
  y = csr_matrix(one_hot_labels)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
    random_state = 13)

  classifier = ClassifierChain(SVC())
  classifier.fit(X_train, y_train)

  preds = classifier.predict(X_test)

  logger.log("Accuracy {}".format(accuracy_score(y_test,preds)))
  logger.log("Hamming score {}".format(hamming_score(y_test, preds)))

  '''
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    https://github.com/scikit-multilearn/scikit-multilearn/issues/84
    https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn
    https://www.kaggle.com/roccoli/multi-label-classification-with-sklearn
    https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
    https://github.com/scikit-multilearn/scikit-multilearn/issues/89
    https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
    https://github.com/mayank408/TFIDF/blob/master/Sklearn%20TFIDF.ipynb

  '''


 

  '''
  X = selected_data.iloc[:, :-1].values
  y = selected_data.iloc[:, -1].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
    random_state = 13)

  classifier = ClassifierChain(LogisticRegression())
  classifier.fit(X_train, y_train)
  '''










