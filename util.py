from features import StructuralFeatures, SentimentFeatures, ContentFeatures

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import itertools


'''
Accuracy for multi-label classification aka Intersection Over Union
'''
def label_based_accuracy(y_true, y_pred):
  
  accuracies = []
  for i in range(y_true.shape[0]):
    y_true_idx = set(y_true[i].nonzero()[0])
    y_pred_idx = set(y_pred[i].nonzero()[0])

    num_correct_labels = len(y_pred_idx.intersection(y_true_idx))
    num_union_labels   = len(y_pred_idx.union(y_true_idx))

    accuracies.append(num_correct_labels/ num_union_labels)

  return np.mean(accuracies)


'''
Computes one-hot representation for each tag from the corpus of all tags 
'''
def get_one_hot_from_str_labels(str_labels):

  atomic_labels_list = [str_label.split() for str_label in str_labels]
  atomic_labels_list = list(itertools.chain.from_iterable(atomic_labels_list))
  atomic_labels_list = np.unique(atomic_labels_list).tolist()

  one_hot_labels = []
  for str_label in str_labels:
    one_hot_labels.append([1 if tag in str_label else 0 for tag in atomic_labels_list])

  return one_hot_labels


'''
Helper for computing and saving to csv files structural, sentiment and content features
'''
def compute_feats(msdialog_dict, logger):

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


'''
Helper for reading structural, sentiment and content features from csv files
'''
def read_feats(logger):

  logger.log("Read structural features ...")
  structural_df = pd.read_csv(logger.get_data_file("structural.csv"))
  logger.log("Read sentiment features ...")
  sentiment_df  = pd.read_csv(logger.get_data_file("sentiment.csv"))
  logger.log("Read content features ...")
  content_df    = pd.read_csv(logger.get_data_file("content.csv"))

  return structural_df, sentiment_df, content_df


'''
Helper for getting train, test and validation shuffled arrays from Data Frame
  train array size is set by train_size in percentage
  validation and test arrays are equal in size
'''
def get_train_test_valid(data_df, train_size):

  one_hot_labels = get_one_hot_from_str_labels(data_df.iloc[:, -1].values)
    
  boolean_dict = {True: 1, False: 0}
  data_df = data_df.replace(boolean_dict)

  X = data_df.iloc[:, :-1].values
  y = csr_matrix(one_hot_labels)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - train_size, 
    random_state = 13)
  X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.5, 
    random_state = 13)

  return X_train, y_train, X_valid, y_valid, X_test, y_test