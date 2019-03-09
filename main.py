from features import StructuralFeatures, SentimentFeatures, ContentFeatures
from data_preproc import DataPreprocessor
from logger import Logger

import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from test import test_svm, test_random_forest, test_ada_boost, test_combiner_svm_randf
from plots import make_tag_occurences_plot

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

  string_labels = feats_df.iloc[:, -1].values
  atomic_labels_list = [str_label.split() for str_label in string_labels]
  atomic_labels_list = list(itertools.chain.from_iterable(atomic_labels_list))
  atomic_labels_list = set(atomic_labels_list)

  one_hot_labels = []
  for string_label in string_labels:
    one_hot_labels.append([1 if tag in string_label else 0 for tag in atomic_labels_list])

  boolean_dict = {True: 1, False: 0}
  feats_df = feats_df.replace(boolean_dict)

  X = feats_df.iloc[:, :-1].values
  y = csr_matrix(one_hot_labels)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
    random_state = 13)

  X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.5, 
    random_state = 13)

  logger.log("Split data in train/validation/test: {}/{}/{}".format(
    X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

  if logger.config_dict['MODE'].lower() == "test":
    if logger.config_dict['COMPUTE_HYPERPARAMS']:
      ada_boost_classifier = test_ada_boost(X_train, y_train, X_test, y_test, 
        logger, X_valid, y_valid)
      random_forest_classifier = test_random_forest(X_train, y_train, X_test, y_test, 
        logger, X_valid, y_valid)
      svm_classifier = test_svm(X_train, y_train, X_test, y_test, logger, 
        X_valid, y_valid)
    else:
      ens_model = test_combiner_svm_randf(X_train, y_train, X_test, y_test, logger)

  elif logger.config_dict['MODE'].lower() == "draw_plots":
    make_tag_occurences_plot(data_preprocessor.occurences_step1, 
      "", "Frequency rank", "Utterance frequency", "occurences_step1.jpg", logger, 
      vertical_line = 37, color = 'white', edgecolor = 'blue')
    make_tag_occurences_plot(data_preprocessor.occurences, 
      "", "Utterance tag", "Frequency", "occurences_final.jpg", logger,
      color = 'blue', plot_tags = True, edgecolor = 'black')

  logger.close()