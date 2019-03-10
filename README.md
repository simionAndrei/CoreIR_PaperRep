# CoreIR Paper Reproduction 
## ["User Intent Prediction in Information-seeking Conversations"](https://arxiv.org/abs/1901.03489)

Code for Group 23 python implementation of Core IR project IN4325.

## Data

For training and testing the models, we used the "MSDialog-Intent" dataset that can be obtain by following `Instructions on Getting the Data` from [Center for intelligent information retrieval](https://ciir.cs.umass.edu/downloads/msdialog/).

## Project structure

The project tree is displayed bellow:
```
root
│   
│   main.py
|   data_preproc.py
|   features.py
|   test.py
|   test_fcn.py
|   util.py
|   plots.py
|   logger.py
|   config.txt
|   
└───data
│   │   content.csv
│   │   features.csv
|   |   sentiment.csv
│   │   dialog_dataset.csv
│   └───
|
└───logs
|   |   Log files in HTML format from hyperpars search, training and testing simple and ensembles models
|   └───
|
└───models
│   │   Saved hyperparamaters after randomize grid search for SVM, AdaBoost and RandomForest
│   |  
│   │   Saved trainined ensmbled models as pickle files (NOT pushed to GitHub due to space issue) 
│   └───
|
└───output
    │   Plots with utterance tags distribution after first and final preprocessing
    │   
    │   Plot with models results for best performance in terms of Accuracy and F1 (Recall and Precision)
    |
    |   results.csv
    └───
```
    
## Config file

```
{
	"DATA_FILE": "MSDialog-Intent.json",   @Original dialog dataset
	"COMPUTE_FEATS": 0,                    @Values: [0, 1] compute features or read features from csv 
	"COMPUTE_HYPERPARAMS": 0,              @Values: [0, 1] compute hyperamas or read the saved ones
	"MODE": "draw_plots",                  @Values: ["test", "draw_plots"] for testing or generating plots
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

The tests for the machine learning models can be run from `main.py` after setting the desired values to `config.txt` and verify the right method from `test.py` is called in `main.py`:
```shell
(core_ir) python main.py
```

The tests for the fully connected neural network model can be run from `test_fcn.py`, after making you have a working [CUDA enviroment](https://www.tensorflow.org/install/gpu) for GPU support:
```shell
(keras_gpu) python test_fcn.py
```
