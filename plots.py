import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
    plt.text(vertical_line + 2, 1500,'X=37', rotation=0)

  plt.savefig(logger.get_output_file(filename), dpi = 120)
  plt.close()


def make_accuracy_f1_plot(results_filename, plot_filename, logger):

  fig = plt.figure(figsize=(6, 6))
  sns.set()

  results_df = pd.read_csv(logger.get_output_file(results_filename))

  models = results_df['Model'].values
  acc = results_df['Accuracy'].values
  f1 = results_df['F1'].values

  colors = ['orange', 'blue', 'red', 'yellow', 'navy', 'maroon', 'green', 'pink', 'teal']
  colors = ['blue', 'blue', 'blue', 'red', 'green', 'orange', 'dimgray', 'maroon', 'navy']
  markers = ['*', 'p', 'D', 'X', 'X', 'X', 'X', 'X', 'X']
  #colors = sns.color_palette("Paired", f1.shape[0])
  print(colors)
  for i, model in enumerate(models):
    x = acc[i]
    y = f1[i]
    plt.scatter(x, y, marker = markers[i], color = colors[i], label = str(model), s = [80])

  plt.xlabel("Accuracy")
  plt.ylabel("F1")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

  plt.savefig(logger.get_output_file(plot_filename), dpi = 120,  bbox_inches='tight')
  plt.close()