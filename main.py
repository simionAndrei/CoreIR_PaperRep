from data_preproc import DataPreprocessor
from logger import Logger

from test import test_svm, test_random_forest, test_ada_boost
from test import test_combiner_svm_randf, test_combiner_ada_randf, test_combiner_svm_ada
from plots import make_tag_occurences_plot, make_accuracy_f1_plot
from util import get_train_test_valid, read_feats, compute_feats

from feature_importance import FeatureImportanceAnalyzer

import pandas as pd

if __name__ == '__main__':
  logger = Logger(show = True, html_output = True, config_file = "config.txt")

  data_preprocessor = DataPreprocessor(logger.config_dict['DATA_FILE'], logger)
  msdialog_dict = data_preprocessor.get_preprocess_data(0.9)

  if logger.config_dict['COMPUTE_FEATS']:
    structural_df, sentiment_df, content_df = compute_feats(msdialog_dict, logger)
  else:
    structural_df, sentiment_df, content_df = read_feats(logger)
        
  if logger.config_dict['MODE'].lower() == "tests":

    feats_df = pd.concat([structural_df, sentiment_df, content_df], axis = 1)
    feats_df['label'] = data_preprocessor.final_tags
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_train_test_valid(
      feats_df, train_size = 0.9) 

    logger.log("Split data in train/validation/test: {}/{}/{}".format(
      X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

    if logger.config_dict['COMPUTE_HYPERPARAMS']:
      ada_boost_classifier = test_ada_boost(X_train, y_train, X_test, y_test, 
        logger, X_valid, y_valid)
      random_forest_classifier = test_random_forest(X_train, y_train, X_test, y_test, 
        logger, X_valid, y_valid)
      svm_classifier = test_svm(X_train, y_train, X_test, y_test, logger, 
        X_valid, y_valid)
    else:
      ens_model = test_combiner_ada_randf(X_train, y_train, X_test, y_test, logger)

  elif logger.config_dict['MODE'].lower() == "plots":
    make_accuracy_f1_plot("best.csv", "acc_f1.jpg", logger)
    make_tag_occurences_plot(data_preprocessor.occurences_step1, 
      "", "Frequency rank", "Utterance frequency", "occurences_step1.jpg", logger, 
      vertical_line = 37, color = 'blue')
    make_tag_occurences_plot(data_preprocessor.occurences, 
      "", "Utterance tag", "Frequency", "occurences_final.jpg", logger,
      color = 'blue', plot_tags = True, edgecolor = 'black')
    make_feats_importance_barplot("feats_imp.csv", "feats_imp.jpg", num_feats_to_plot = 10, 
      logger = logger)

  elif logger.config_dict['MODE'].lower() == "feats":
    analyzer = FeatureImportanceAnalyzer(sentiment_df, content_df, structural_df, 
      data_preprocessor.final_tags, logger)
    analyzer.analyze_sentiment()
    analyzer.analyze_content()
    analyzer.analyze_structural()
    analyzer.analyze_combinations()
    analyzer.analyze_individual_importance()

  logger.close()