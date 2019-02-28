from logger import Logger
from data_preproc import DataPreprocessor

if __name__ == '__main__':
  logger = Logger(show = True, html_output = True)

  data_preprocessor = DataPreprocessor('MSDialog-Intent.json', logger)
  msdialog_dict = data_preprocessor.get_preprocess_data(0.9)