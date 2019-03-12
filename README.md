# CoreIR Paper Reproduction 
## ["User Intent Prediction in Information-seeking Conversations"](https://arxiv.org/abs/1901.03489)

Code for Group 26 python implementation of Core IR project IN4325.

Team members:
 * [Simion-Constantinescu Andrei](https://www.linkedin.com/in/andrei-simion-constantinescu/)
 * [Nele Albers](https://www.tudelft.nl/ewi/)
 * [Priya Sarkar](https://www.tudelft.nl/ewi/)

## Data

For training and testing the models, we used the "MSDialog-Intent" dataset that can be obtained by following the `Instructions on Getting the Data` from the [Center for intelligent information retrieval](https://ciir.cs.umass.edu/downloads/msdialog/).

## Project structure

The project tree is displayed bellow:
```
root
│   
│   main.py			application entry point
|   data_preproc.py		performes preprocessing as original paper describes
|   features.py			computes sentiment, content and structural features
|   test.py			run hyperparams grid-search on validation and test performance of simple and combined models
|   test_fcn.py			test performance of fully-connected neural model 96N-Dropout50-48N-12Sigmoid
|   util.py			utility functions like computing label based accuracy, read features, transform to one-hot a.s.o
|   feature_importance.py	compute feature importance in Groups (sentiment, content, strcutural and combinations) and individually
|   plots.py			generate barplots for occurrences tags distribution and Accuracy vs F1 plot
|   logger.py			logging system for generating folders initial structure and saving application logs to html files  
|   config.txt			application configuration file 
|   
└───data
│   │   content.csv
│   │   features.csv
|   |   sentiment.csv
│   │   dialog_dataset.csv
│   └───
|
└───logs
|   |   Log files in HTML format for 
|   |   	hyperpars search, training and testing simple and ensembles models, compute importance of features
|   └───
|
└───models
│   │   Saved hyperparamaters after randomize grid search for SVM, AdaBoost and RandomForest
│   |  
│   │   Saved trained ensmbled models as pickle files (NOT pushed to GitHub due to space issue) 
│   └───
|
└───output
    │   Plots with utterance tags distribution after first and final preprocessing
    │   
    │   Plot with models results for best performance in terms of Accuracy and F1 (Recall and Precision)
    |
    |   results.csv
    |
    |   best.csv
    |
    |   feats_imp.csv
    └───
```
    
## Config file

```
{
	"DATA_FILE": "MSDialog-Intent.json",   @Original dialog dataset
	"COMPUTE_FEATS": 0,                    @Values: [0, 1] compute features or read features from csv 
	"COMPUTE_HYPERPARAMS": 0,              @Values: [0, 1] compute hyperamas or read the saved ones
	"MODE": "draw_plots",                  @Values: ["tests", "plots", "feats"] for model tests/plots/features-importance
	"BEST_SVM": "SVC_params_2019-03-08_13_50_23.json",
	"BEST_ADA": "AdaBoostClassifier_params_2019-03-08_17_28_49.json",
	"BEST_RANDF": "RandomForestClassifier_params_2019-03-08_12_26_28.json",
	"ENS1_WEIGHTS": [0.4, 0.6],            @Ensemble 1 weights 0.4SVM + 0.6Ada
	"ENS2_WEIGHTS": [0.5, 0.5],            @Ensemble 2 weights 0.5SVM + 0.5RandF
	"ENS3_WEIGHTS": [0.4, 0.6]             @Ensemble 3 weights 0.4Ada + 0.6RandF
}
```

## Environment
The application can be run in [Anaconda](https://www.anaconda.com/download/) Windows environment.

The Anaconda enviroments needs to be created `core_ir` and `keras_gpu`.

For creating and setting `core_ir` enviroment:
```shell
(base) conda create -n core_ir python=3.7 anaconda
(base) conda activate core_ir
(core_ir) conda install -c anaconda nltk
(core_ir) pip install scikit-multilearn
(core_ir) pip install vaderSentiment
```

We will also need to download `punkt` and `opinion_lexicon` from `nltk`:
```python
import nltk
nltk.download('punkt')
nltk.download('opinion_lexicon')
```

For creating and setting `keras_gpu` enviroment:
```shell
(base) conda create -n keras_gpu python=3.6 anaconda
(base) conda activate keras_gpu
(keras_gpu) pip install tensorflow-gpu
(keras_gpu) conda install -c conda-forge keras 
```

## Run
To run `main.py`, we need to obtain `MSDialog.json` dataset by requesting access following the instruction from [Data section](#data).
After getting `MSDialog-Intent.json`, copy it in the `data` folder. 

The tests for the machine learning models can be run from `main.py` after setting the desired values to `config.txt` and verifying the right method from `test.py` is called in `main.py`:
```shell
(core_ir) python main.py
```

The tests for the fully connected neural network model can be run from `test_fcn.py`, after making sure you have a working [CUDA enviroment](https://www.tensorflow.org/install/gpu) for GPU support:
```shell
(keras_gpu) python test_fcn.py
```
