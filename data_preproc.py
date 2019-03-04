from collections import Counter
import numpy as np
import itertools
import json
import random
import re


class DataPreprocessor():

  def __init__(self, data_filename, logger):
    self.logger = logger
    self.data_filename = data_filename
    self._read_data()
    self._compute_stats()

  def _read_data(self):
    with open(self.logger.get_data_file(self.data_filename)) as fp:
      self.msdialog_dict = json.load(fp)
    self.logger.log("Finished read data from {}".format(self.data_filename))

  def _compute_ignored_tags(self):
    self.ignored_labels = ['GG', 'JK', 'O']
    self.ignored_combinations  = [item[0] + ' ' + item[1] for item in itertools.combinations(
      self.ignored_labels, 2)]
    self.ignored_combinations += [item[1] + ' ' + item[0] for item in itertools.combinations(
      self.ignored_labels, 2)]


  def _compute_stats(self):
    self.logger.log("Number of dialogs {}".format(
      np.unique(list(self.msdialog_dict.keys())).shape[0]))

    num_utterances = sum([len(item['utterances']) for item in self.msdialog_dict.values()])
    self.logger.log("Number of utterances {}".format(num_utterances))

    self.logger.log("Randomly selected final utterance {}".format(
      self.msdialog_dict[np.random.choice(list(self.msdialog_dict.keys()))]['utterances'][-1]))


  def _compute_initial_tags(self):
    self.initial_tags = []
    for item in self.msdialog_dict.values():
      crt_tags = [' '.join(utterance['tags'].split()) for utterance in item['utterances']]
      self.initial_tags += crt_tags

    self.logger.log("Initial number of unique tags {}".format(len(set(self.initial_tags))))


  def _change_tags(self):

    if set(self.initial_tags) & set(self.ignored_combinations):
      self.logger.log("Data contains combination of ignored labels: e.g {}".format(
        random.sample(set(self.initial_tags) & set(self.ignored_combinations), 1)))

      num_occurences_ignored_combinations = len(
        [tag for tag in self.initial_tags if tag in self.ignored_combinations])
      self.logger.log("Number of this combinations is {}".format(
        num_occurences_ignored_combinations))

    replace_regex = re.compile('|'.join(map(re.escape, self.ignored_labels)))
    for item_key in self.msdialog_dict.keys():
      dialog = self.msdialog_dict[item_key]['utterances']
      for i, qa in enumerate(dialog):
        new_tag = ' '.join(qa['tags'].split())
        if any(label in new_tag for label in self.ignored_labels) and ' ' in new_tag and new_tag not in self.ignored_combinations:
          new_tag = replace_regex.sub("", new_tag)

        self.msdialog_dict[item_key]['utterances'][i]['tags'] = ' '.join(new_tag.split())

    self.final_tags = [utterance['tags'] for item in self.msdialog_dict.values() for utterance in item['utterances']]
    self.logger.log("Number of unique tags after first preprocess step {}".format(len(set(self.final_tags))))


  def _select_topNp_tags(self, n):
    occurences = dict(Counter(self.final_tags)).items()
    self.occurences = sorted(occurences, key=lambda tup: tup[1], reverse = True)

    total_num_occurences = sum([item[1] for item in occurences])
    for top_Np_idx in range(len(occurences)):
      if sum([item[1] for item in self.occurences[:top_Np_idx]]) / total_num_occurences > n:
        break

    self.selected_tags = [item[0] for item in self.occurences[:top_Np_idx]]
    self.logger.log("Number of unique tags that made {:.2f}% of the occurences {}".format(
      n * 100, top_Np_idx))
    self.logger.log("{}".format(self.selected_tags))


  def get_preprocess_data(self, topNp):

    self._compute_initial_tags()
    self._compute_ignored_tags()
    self._change_tags()
    self._select_topNp_tags(topNp)

    return self.msdialog_dict

if __name__=='__main__':
  print("Library module. No main function")