from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.models import Model

from logger import Logger

import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from util import label_based_accuracy

def fcn(input_size, feats_size):
  X_input = Input(shape = (input_size,), name='Input')
   
  X = Dense(feats_size, activation = 'selu', name = 'Hidden_1')(X_input)
  X = Dropout(rate = 0.5, name='Dropout_half1')(X)
  X = Dense(int(feats_size / 2), activation = 'selu', name = 'Hidden_2')(X)
  X = Dense(12, activation = 'sigmoid', name='Output')(X)
   
  model = Model(inputs = X_input, outputs = X, name = "FC_168-84")
  model.compile(optimizer = "adam", loss = "binary_crossentropy", 
                metrics = ['accuracy'])
  return model


logger = Logger(show = True, html_output = True, config_file = "config.txt")

feats_df = pd.read_csv(logger.get_data_file("dialog_dataset.csv"))
string_labels = feats_df.iloc[:, -1].values
atomic_labels_list = [str_label.split() for str_label in string_labels]
atomic_labels_list = list(itertools.chain.from_iterable(atomic_labels_list))
atomic_labels_list = set(atomic_labels_list)

one_hot_labels = []
for string_label in string_labels:
  one_hot_labels.append([1 if tag in string_label else 0 for tag in atomic_labels_list])

boolean_dict = {True: 1, False: 0}
feats_df = feats_df.replace(boolean_dict)

X = feats_df.iloc[:, :-1].values
y = np.array(one_hot_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
  random_state = 13)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = 0.5, 
    random_state = 13)

std_scale = StandardScaler().fit(X_train)

X_train_std = std_scale.transform(X_train)
X_test_std  = std_scale.transform(X_test)
X_valid_std = std_scale.transform(X_valid)

model = fcn(X_train.shape[1],  X_train.shape[1] * 4)
model.summary()

callbacks = []
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10))
model.fit(x = X_train_std, y = y_train, batch_size = 64, epochs = 200, 
  validation_data = (X_valid_std, y_valid), callbacks = callbacks)

yhat = model.predict(X_test_std)
y_pred = yhat > 0.5

accuracy = label_based_accuracy(y_test, y_pred)
logger.log("Accuracy label based score {}".format(accuracy))
logger.log("Subset accuracy {}".format(accuracy_score(y_test, y_pred)))
logger.log("Recall {}".format(recall_score(y_test, y_pred, average = 'micro')))
logger.log("Precision {}".format(precision_score(y_test, y_pred, average = 'micro')))
logger.log("F1 {}".format(f1_score(y_test, y_pred, average = 'micro')))