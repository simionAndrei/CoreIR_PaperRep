from features import StructuralFeatures, SentimentFeatures, ContentFeatures
from data_preproc import DataPreprocessor
from logger import Logger


if __name__ == '__main__':
  logger = Logger(show = True, html_output = True)

  data_preprocessor = DataPreprocessor('MSDialog-Intent.json', logger)
  msdialog_dict = data_preprocessor.get_preprocess_data(0.9)

  structural_feats_extractor = StructuralFeatures(logger)
  structural_df = structural_feats_extractor.compute_features(msdialog_dict)
  logger.log("Structural Features: \n {}".format(structural_df.head()))

  sentiment_feats_extractor = SentimentFeatures(logger)
  sentiment_df = sentiment_feats_extractor.compute_features(msdialog_dict)
  logger.log("Sentiment Features: \n {}".format(sentiment_df.head()))

  '''
  content_feats_extractor = ContentFeatures(logger)
  content_df = content_feats_extractor.compute_features(msdialog_dict)
  logger.log("Content Features: \n {}".format(content_df.head()))
  '''