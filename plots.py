import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.rcParams['font.weight']= 'bold'

def make_tag_occurences_plot(occurences, plt_title, x_label, y_label, filename, logger, 
  plot_tags = False, vertical_line = None, color = 'blue', edgecolor = False, log = False):

  fig = plt.figure(figsize=(8, 8))
  sns.set()
  plt.bar(range(len(occurences)), [item[1] for item in occurences], width = 0.9, 
    color = color, edgecolor = edgecolor, log = log)

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(plt_title)

  if plot_tags:
    x = np.array(range(len(occurences)))
    my_xticks = [item[0] for item in occurences]
    plt.xticks(x, my_xticks, rotation = 90)

  if vertical_line is not None:
    plt.axvline(x = vertical_line, color = 'black')
    plt.text(vertical_line + 2, 1500,'X=37',rotation=0)

  plt.savefig(logger.get_output_file(filename), dpi = 120)
  plt.close()