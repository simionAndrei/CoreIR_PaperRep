from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

import pandas as pd
from tqdm import tqdm


from abc import ABC, abstractmethod
class AbstractFeatures(ABC):

  def __init__(self, logger):
    self.logger = logger

  @abstractmethod
  def compute_features(self, dialog_dict):
    pass



class SentimentFeatures(AbstractFeatures):

  def __init__(self, logger):
    super().__init__(logger)

  def _get_feature_per_sentence(self, dialog_part):
    sentence = dialog_part['utterance']

    vader_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = vader_analyzer.polarity_scores(sentence)

    has_thank = "thank" in sentence.lower()
    has_exclamation = "!" in sentence
    has_feedback = "did not" in sentence.lower() or "does not" in sentence.lower()
    pos_opinion_count = [word for word in word_tokenize(sentence) if word in opinion_lexicon.positive()]
    neg_opinion_count = [word for word in word_tokenize(sentence) if word in opinion_lexicon.negative()]

    return [has_thank, has_exclamation, has_feedback, 
            sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'],
            pos_opinion_count, neg_opinion_count]


  def compute_features(self, dialog_dict):

    self.logger.log("Start computing sentiment features...")
    sentiment_features = []
    for _, dialog in tqdm(dialog_dict.items()):
      for dialog_part in dialog['utterances']:
        crt_feats = self._get_feature_per_sentence(dialog_part)
        structural_features.append(crt_feats)

    colnames = ["Thank", "Exclamation_Mark", "Feedback", 
                "Sentiment_Scores_NEG", "Sentiment_Scores_NEU", "Sentiment_Scores_POS",
                "Opinion_Lexicon_POS", "Opinion_Lexicon_NEG"]
    self.logger.log("Finished computing sentiment features")

    return pd.DataFrame(sentiment_features, columns = colnames)



class StructuralFeatures(AbstractFeatures):

  def __init__(self, logger):
    super().__init__(logger)


  def _get_feature_per_sentence(self, dialog_part, dialog_starter_id, num_dialog_lines):
    sentence = dialog_part['utterance']
    true_words = [word for word in word_tokenize(sentence) if word not in stopwords.words('english')]

    stemmer = SnowballStemmer('english')
    true_stemmed_words = [stemmer.stem(word) for word in true_words]

    pos_in_dialog = dialog_part['utterance_pos']
    norm_pos_in_dialog = pos_in_dialog / num_dialog_lines
    num_words = len(true_words)
    num_unique_words = len(set(true_words))
    num_unique_words_after_stemming = len(set(true_stemmed_words))
    is_starter = dialog_part['user_id'] == dialog_starter_id

    return [pos_in_dialog, norm_pos_in_dialog, num_words, num_unique_words, 
            num_unique_words_after_stemming, is_starter]


  def compute_features(self, dialog_dict):

    self.logger.log("Start computing structural features...")
    structural_features = []
    for _, dialog in tqdm(dialog_dict.items()):
      dialog_starter_id = dialog['utterances'][0]['user_id']
      num_dialog_lines = len(dialog['utterances'])
      for dialog_part in dialog['utterances']:
        crt_feats = self._get_feature_per_sentence(dialog_part, dialog_starter_id,
          num_dialog_lines)
        structural_features.append(crt_feats)

    colnames = ["Absolute_Position", "Normalized_Position", "Utterance_Length",
                "Utterance_Length_Unique", "Utterance_Length_Stemmed_Unique", 
                "Is_Starter"]
    self.logger.log("Finished computing structural features")

    return pd.DataFrame(structural_features, columns = colnames)



class ContentFeatures(AbstractFeatures):

  def __init__(self, logger):
    super().__init__(logger)

  def compute_features(self, dialog_dict):
    pass



if __name__=='__main__':
  print("Library module. No main function")