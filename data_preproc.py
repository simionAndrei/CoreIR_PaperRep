import numpy as np
import itertools
import json


class DataPreprocessor():

	def __init__(data_filename, logger):
		self.data_filename = data_filename
		self._read_data()


	def _read_data(self):
		with open(self.logger.get_data_file(self.data_filename)) as fp:
    		self.msdialog_dict = json.load(fp)


    def _compute_ignored_tags(self):
    	self.ignored_labels = ['GG', 'JK', 'O']
    	self.ignored_combinations  = [item[0] + ' ' + item[1] for item in itertools.combinations(ignored_labels, 2)]
    	self.ignored_combinations += [item[1] + ' ' + item[0] for item in itertools.combinations(ignored_labels, 2)]


    def _compute_stats(self):
    	self.logger.log("Number of dialogs {}".format(
    		np.unique(list(self.msdialog_dict.keys())).shape[0]))

    	num_utterances = sum([len(item['utterances']) for item in self.msdialog_dict.values()])
		self.logger.log("Number of utterances {}".format(num_utterances))

		self.logger.log("Randomly selected final utterance {}".format(
			msdialog_dict[np.random.choice(list(self.msdialog_dict.keys()))]['utterances'][-1]))


	def _compute_initial_tags(self):
		self.initial_tags = []
		for item in self.msdialog_dict.values():
    		crt_tags = [' '.join(utterance['tags'].split()) for utterance in item['utterances']]
    		self.initial_tags += crt_tags

    	self.logger.log("Initial number of unique tags {}".format(len(set(self.initial_tags))))


    def _change_tags(self):

    	if set(self.initial_tags) & set(self.ignored_combinations):
    		self.logger.log("Data contains combination of ignored labels: e.g {}".format(
                random.sample(set(initial_tags) & set(ignored_combinations), 1)))

    num_occurences_ignored_combinations = len([tag for tag in self.initial_tags if tag in self.ignored_combinations])
    self.logger.log("Number of this combinations is {}".format(num_occurences_ignored_combinations))

for item_key in msdialog_dict.keys():
    dialog = msdialog_dict[item_key]['utterances']
    for i, qa in enumerate(dialog):
        new_tag = ' '.join(qa['tags'].split())
        if any(label in new_tag for label in ignored_labels) and ' ' in new_tag and new_tag not in ignored_combinations:
            for label in ignored_labels:
                new_tag = new_tag.replace(label, "")
                
        msdialog_dict[item_key]['utterances'][i]['tags'] = ' '.join(new_tag.split())

all_tags = [utterance['tags'] for item in msdialog_dict.values() for utterance in item['utterances']]
logger.log("Number of unique tags after first preprocess step {}".format(len(set(all_tags))))



if __name__=='__main__':
  print("Library module. No main function")