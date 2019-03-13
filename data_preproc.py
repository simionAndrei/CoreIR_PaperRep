from collections import Counter
import numpy as np
import itertools
import json
import random
import re


'''
Class for preprocessing MSDialog Intent Utterances Datataset

'''
class DataPreprocessor():

  def __init__(self, data_filename, logger):
    self.logger = logger
    self.data_filename = data_filename
    self._read_data()
    self._compute_stats()


  '''
  Read MSDialogIntent.json file
  '''
  def _read_data(self):
    with open(self.logger.get_data_file(self.data_filename)) as fp:
      self.msdialog_dict = json.load(fp)
    self.logger.log("Finished read data from {}".format(self.data_filename), show_time = True)


  '''
  Stores ignored labels and each combination of 2 ignored labels
  '''
  def _compute_ignored_tags(self):
    self.ignored_labels = ['GG', 'JK', 'O']
    self.ignored_combinations  = [item[0] + ' ' + item[1] for item in itertools.combinations(
      self.ignored_labels, 2)]
    self.ignored_combinations += [item[1] + ' ' + item[0] for item in itertools.combinations(
      self.ignored_labels, 2)]


  '''
  Compute number of dialogs and total number of utterances
  '''
  def _compute_stats(self):
    self.logger.log("Number of dialogs {}".format(
      np.unique(list(self.msdialog_dict.keys())).shape[0]))

    num_utterances = sum([len(item['utterances']) for item in self.msdialog_dict.values()])
    self.logger.log("Number of utterances {}".format(num_utterances))

    self.logger.log("Randomly selected final utterance {}".format(
      self.msdialog_dict[np.random.choice(list(self.msdialog_dict.keys()))]['utterances'][-1]))

  '''
  Computes initial tags distribution
  '''
  def _compute_initial_tags(self):
    self.initial_tags = []
    for item in self.msdialog_dict.values():
      crt_tags = [' '.join(utterance['tags'].split()) for utterance in item['utterances']]
      self.initial_tags += crt_tags

    self.logger.log("Initial number of unique tags {}".format(len(set(self.initial_tags))))


  '''
  Remove each ignored label from the tag only if the tag ontains another label
  GG+OQ -> OQ
  GG+JK -> unmodified
  GG -> unmodified
  '''
  def _change_tags_remove_irrelevant(self):

    if set(self.initial_tags) & set(self.ignored_combinations):
      self.logger.log("Data contains combination of ignored labels: e.g {}".format(
        random.sample(set(self.initial_tags) & set(self.ignored_combinations), 1)))

      num_occurences_ignored_combinations = len(
        [tag for tag in self.initial_tags if tag in self.ignored_combinations])
      self.logger.log("Number of this combinations is {}".format(
        num_occurences_ignored_combinations))

    # regex for replacing ignored labels as words
    replace_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, self.ignored_labels)))
    dialog_ids = list(self.msdialog_dict.keys())
    for item_key in dialog_ids:
      dialog = self.msdialog_dict[item_key]['utterances']
      for i, qa in enumerate(dialog):
        new_tag = ' '.join(qa['tags'].split())
        if any(label in new_tag for label in self.ignored_labels) and ' ' in new_tag and new_tag not in self.ignored_combinations:
          new_tag = replace_regex.sub("", new_tag)

        self.msdialog_dict[item_key]['utterances'][i]['tags'] = ' '.join(new_tag.split())

    self.tags_after_step1 = [utterance['tags'] for item in self.msdialog_dict.values(
      ) for utterance in item['utterances']]
    self.logger.log("Number of unique tags after first preprocess step {}".format(
      len(set(self.tags_after_step1))))


  '''
  Change tags by replacing rare tags with only one randomly selected label 
  for the (1-N)% less frequent tags
  '''
  def _change_tags_remove_rare(self):

    self.selected_tags.append('O')
    dialog_ids = list(self.msdialog_dict.keys())
    for item_key in dialog_ids:
      dialog = self.msdialog_dict[item_key]['utterances']
      for i, qa in enumerate(dialog):
        crt_tag = qa['tags']
        if crt_tag not in self.selected_tags:
          crt_tag = np.random.choice(crt_tag.split())
          self.msdialog_dict[item_key]['utterances'][i]['tags'] = crt_tag

    self.final_tags = [utterance['tags'] for item in self.msdialog_dict.values(
      ) for utterance in item['utterances']]
    self.logger.log("Number of unique tags after second preprocess step {}".format(
      len(set(self.final_tags))))
    self.logger.log("{}".format(self.selected_tags))


  '''
  Compute the most frequent tags that made N% of the total number of occurences
  '''
  def _select_topNp_tags(self, n):
    occurences = dict(Counter(self.tags_after_step1)).items()
    self.occurences_step1 = sorted(occurences, key=lambda tup: tup[1], reverse = True)

    total_num_occurences = sum([item[1] for item in occurences])
    for top_Np_idx in range(len(occurences)):
      if sum([item[1] for item in self.occurences_step1[:top_Np_idx]]) / total_num_occurences > n:
        break

    self.selected_tags = [item[0] for item in self.occurences_step1[:top_Np_idx]]
    self.logger.log("Number of unique tags that made {:.2f}% of the occurences {}".format(
      n * 100, top_Np_idx))
    self.logger.log("{}".format(self.selected_tags))


  '''
  Public method for getting preprocessed data based on the selected topNp% occurences
  '''
  def get_preprocess_data(self, topNp):

    np.random.seed(0)
    self._compute_initial_tags()
    self._compute_ignored_tags()
    self._change_tags_remove_irrelevant()
    self._select_topNp_tags(topNp)
    self._change_tags_remove_rare()

    occurences = dict(Counter(self.final_tags)).items()
    self.occurences = sorted(occurences, key=lambda tup: tup[1], reverse = True)
    
    self.logger.log("Final tags distribution \n {}".format(self.occurences))

    return self.msdialog_dict

if __name__=='__main__':
  print("Library module. No main function")