from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

import pandas as pd
from tqdm import tqdm
import string

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



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
    sentence = sentence.translate(str.maketrans('','',string.punctuation))
    pos_opinion_count = len(set(opinion_lexicon.positive()) & set(sentence.split()))
    neg_opinion_count = len(set(opinion_lexicon.negative()) & set(sentence.split()))
    
    return [has_thank, has_exclamation, has_feedback, 
            sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'],
            pos_opinion_count, neg_opinion_count]


  def compute_features(self, dialog_dict):

    self.logger.log("Start computing sentiment features...")
    sentiment_features = []
    for _, dialog in tqdm(dialog_dict.items()):
      for dialog_part in dialog['utterances']:
        crt_feats = self._get_feature_per_sentence(dialog_part)
        sentiment_features.append(crt_feats)

    colnames = ["Thank", "Exclamation_Mark", "Feedback", 
                "Sentiment_Scores_NEG", "Sentiment_Scores_NEU", "Sentiment_Scores_POS",
                "Opinion_Lexicon_POS", "Opinion_Lexicon_NEG"]
    self.logger.log("Finished computing sentiment features", show_time = True)

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
    self.logger.log("Finished computing structural features", show_time = True)

    return pd.DataFrame(structural_features, columns = colnames)



class ContentFeatures(AbstractFeatures):

  def __init__(self, logger):
    super().__init__(logger)


  def _get_feature_per_sentence(self, dialog_part, all_parts):

    has_question = "?" in dialog_part

    all_parts = [item.translate(str.maketrans('','',string.punctuation)) for item in all_parts]
    dialog_part = dialog_part.translate(str.maketrans('','',string.punctuation))
    other_parts = [item for item in all_parts if item != dialog_part]

    vectorizer1 = TfidfVectorizer()
    tfid_all_dialog = vectorizer1.fit_transform(
      [dialog_part] + all_parts)

    vectorizer2 = TfidfVectorizer()
    tfid_current_rest_joined = vectorizer2.fit_transform(
      [dialog_part, " ".join(other_parts)])

    similarity_with_initial_part = float(
      cosine_similarity(tfid_all_dialog[0], tfid_all_dialog[1]))
    similarity_with_all_parts_joined = float(
      cosine_similarity(tfid_current_rest_joined[0], tfid_current_rest_joined[1]))

    has_duplicate = "same" in dialog_part or "similar" in dialog_part
    list_5w = ["what", "where", "when", "why", "who", "how"]
    one_hot_5w = [1 if word in dialog_part.lower() else 0 for word in list_5w]

    return [similarity_with_initial_part, similarity_with_all_parts_joined,
            has_question, has_duplicate] + one_hot_5w


  def compute_features(self, dialog_dict):

    self.logger.log("Start computing content features...")
    content_features = []
    for _, dialog in tqdm(dialog_dict.items()):
      all_parts = [item['utterance'] for item in dialog['utterances']]
      for dialog_part in dialog['utterances']:
        crt_feats = self._get_feature_per_sentence(dialog_part['utterance'],
          all_parts)
        content_features.append(crt_feats)

    colnames = ["Initial_Utterance_Similarity", "Dialog_Similarity", "Question_Mark",
                "Duplicate", "What", "Where", "When", "Why", "Who", "How"]
    self.logger.log("Finished computing content features", show_time = True)

    return pd.DataFrame(content_features, columns = colnames)



if __name__=='__main__':
  print("Library module. No main function")