import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import gridspec

plt.rc('font', weight='bold')        
plt.rc('axes', labelweight='bold')
plt.rcParams.update({'font.size': 16})

def make_tag_occurences_plot(occurences, plt_title, x_label, y_label, filename, logger, 
  plot_tags = False, vertical_line = None, color = 'blue', edgecolor = None, log = False):

  if plot_tags:
    fig = plt.figure(figsize=(10, 10))
  else:
    fig = plt.figure(figsize=(12, 12))

  sns.set()

  if plot_tags:
    plt.bar(range(len(occurences)), [item[1] for item in occurences], width = 0.9, 
      color = color, edgecolor = edgecolor, log = log)
  else:
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4]) 
    ax0 = plt.subplot(gs[0])
    ax0.bar(range(len(occurences)), [item[1] for item in occurences], width = 0.9, 
      color = 'white', edgecolor = 'blue', log = False)
    ax0.axvline(x = vertical_line, color = 'black')
    ax1 = plt.subplot(gs[1])
    ax1.bar(range(len(occurences)), [item[1] for item in occurences], width = 1.0, 
      color = color, edgecolor = 'white', log = True)

  plt.xlabel(x_label, fontsize = 18)
  plt.ylabel(y_label, fontsize = 18)
  plt.title(plt_title)
  plt.xticks(fontsize = 15)
  plt.yticks(fontsize = 15)

  if plot_tags:
    x = np.array(range(len(occurences)))
    my_xticks = [item[0] for item in occurences]
    plt.xticks(x, my_xticks, rotation = 90, fontsize = 15)

  if vertical_line is not None:
    plt.axvline(x = vertical_line, color = 'black')
    plt.text(vertical_line + 2, 1500,'X=37', rotation=0)

  plt.savefig(logger.get_output_file(filename), dpi = 120, bbox_inches='tight')
  plt.close()


def make_accuracy_f1_plot(results_filename, plot_filename, logger):

  fig = plt.figure(figsize=(7, 7))
  sns.set()

  results_df = pd.read_csv(logger.get_output_file(results_filename))

  models = results_df['Model'].values
  acc = results_df['Accuracy'].values
  f1 = results_df['F1'].values

  colors = ['blue', 'blue', 'blue', 'red', 'green', 'orange', 'dimgray', 'maroon', 'navy']
  markers = ['*', 'p', 'D', 'X', 'X', 'X', 'X', 'X', 'X']
  for i, model in enumerate(models):
    x = acc[i]
    y = f1[i]
    plt.scatter(x, y, marker = markers[i], color = colors[i], label = str(model), s = [80])

  plt.xlabel("Accuracy", fontsize = 16)
  plt.ylabel("F1", fontsize = 16)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
    fancybox=True, shadow=True, fontsize = 15)
  plt.xticks(fontsize = 13)
  plt.yticks(fontsize = 13)

  plt.savefig(logger.get_output_file(plot_filename), dpi = 120,  bbox_inches='tight')
  plt.close()


def make_feats_importance_barplot(feats_imp_filename, plot_filename, 
  num_feats_to_plot, logger):

  fig = plt.figure(figsize=(8, 10))
  sns.set()

  df = pd.read_csv(logger.get_output_file(feats_imp_filename))

  model1_feats_imp_scores = df.iloc[0, 1:]
  model2_feats_imp_scores = df.iloc[1, 1:]

  model1_name_score_pairs = list(zip(df.columns[1:], model1_feats_imp_scores))
  model2_name_score_pairs = list(zip(df.columns[1:], model2_feats_imp_scores))

  model1_name_score_pairs = sorted(model1_name_score_pairs, key=lambda tup: tup[1], reverse = True)
  model2_name_score_pairs = sorted(model2_name_score_pairs, key=lambda tup: tup[1], reverse = True)

  model2_names, model2_scores = zip(*model2_name_score_pairs)
 
  model1_scores = [dict(model1_name_score_pairs)[name] for name in model2_names]

  model2_names = model2_names[:num_feats_to_plot]

  model1_scores = model1_scores[:num_feats_to_plot]
  model2_scores = model2_scores[:num_feats_to_plot]

  x_range = np.array(range(len(model2_names)))

  plt.yticks(fontsize = 15)
  plt.ylabel("Relative importance score", fontsize = 18)
  plt.bar(x_range, model2_scores, width = 0.4, color = 'red')
  plt.xticks(x_range, model2_names, rotation = 90, fontsize = 16)

  plt.bar(x_range + 0.4, model1_scores, width = 0.4, color = 'blue')
  
  plt.legend(["Randon Forest", "Ada Boost"], fontsize = 16)
  plt.savefig(logger.get_output_file(plot_filename), dpi = 120, fontsize = 16, bbox_inches='tight')
  plt.close()

















