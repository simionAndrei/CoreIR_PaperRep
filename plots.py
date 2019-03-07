import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.rcParams['font.weight']= 'bold'

def make_occurences_plot(occurences, filename, logger, color = 'blue', 
  log = False):

  fig = plt.figure(figsize=(8, 8))
  sns.set()
  plt.bar(range(len(occurences)), [item[1] for item in occurences], width = 0.9, 
    color = color, edgecolor = 'black', log = log)

  plt.xlabel("Utterance tag")
  plt.ylabel("Number of occurences")
  plt.title("Distribution of the unique utterances tags")

  x = np.array(range(len(occurences)))
  my_xticks = [item[0] for item in occurences]
  plt.xticks(x, my_xticks, rotation = 90)
  plt.savefig(logger.get_output_file(filename), dpi = 120)
  plt.close()