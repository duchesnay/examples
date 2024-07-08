#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:43:20 2023

@author: edouard.duchenay@cea.fr
"""

###############################################################################
# %% Imports

import numpy as np
import os
import os.path
import numpy as np
import pandas as pd
import nibabel
import glob
import re

# Train test split
from sklearn.model_selection import train_test_split

# Models
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Metrics
import sklearn.metrics as metrics

# Resampling
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

###############################################################################
# %% Parameters

BASEDIR = '/home/ed203246/data/psy_sbox'
# BASEDIR = '/neurospin/psy_sbox/'

WD = os.path.join(BASEDIR, 'analyses/2023_localizer_toy-analysis')
N_FOLDS_INNER = 3

# os.makedirs(WD)


###############################################################################
# %% Utils function

def cv_train_test_scores_params(model, y_train, y_pred_train, y_test,
                                y_pred_test):
    """Compute CV score, train and test score from a cv grid search model.

    Parameters
    ----------
    model : Pipeline or GridSearchCV
        Model. If Pipeline, assume the predictive model is the last step.
    y_train : array
        True train values.
    y_pred_train : array
        Predicted train values.
    y_test : array
        True test values.
    y_pred_test : array
        Predicted test values.

    Returns
    -------
    info : TYPE
        DataFrame(r2_cv, r2_train, mae_train, mse_train).
    """
    # fetch predictor (second items in the pipeline)
    # If the model is a pipeline pick the last step, ie the predictor
    gridsearchcv = model.steps[-1][1] if hasattr(model, "steps") else model
    # Default estimator’s score method is used: R^2 for regression
    r2_cv = gridsearchcv.best_score_
    # and best params
    best_params = gridsearchcv.best_params_

    # Train scores
    r2_train = metrics.r2_score(y_train, y_pred_train)
    mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
    mse_train = metrics.mean_squared_error(y_train, y_pred_train)

    # Test scores
    r2_test = metrics.r2_score(y_test, y_pred_test)
    mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
    mse_test = metrics.mean_squared_error(y_test, y_pred_test)

    info = pd.DataFrame([[r2_cv, r2_train, mae_train, mse_train,
                          r2_test, mae_test, mse_test,
                          str(best_params)]],
                        columns=("r2_cv", "r2_train", "mae_train", "mse_train",
                                 "r2_test", "mae_test", "mse_test",
                                 "best_params"))

    return info

###############################################################################
# %% Load Data

# List files
img_filenames = glob.glob(os.path.join(BASEDIR, 'localizer/derivatives/cat12-12.6_vbm/sub-*/ses-*/anat/mri/mwp1*.nii'))
regex = re.compile("/sub-([^/]+)/") # match (sub-...)_
participant_ids = [regex.findall(s)[0] for s in img_filenames]
files_df = pd.DataFrame(dict(participant_id=participant_ids, img_filename=img_filenames))
assert files_df.shape == (88, 2)

# Participants
data = pd.read_csv(os.path.join(BASEDIR, 'localizer/participants.tsv'), delimiter='\t')
data = data.dropna()
assert data.shape == (90, 4)

# Merge with files
data = pd.merge(data, files_df)
assert data.shape == (84, 5)

# Read images

mask_filename = os.path.join(WD, 'data/mni_cerebrum-mask.nii.gz')
mask_arr = nibabel.load(mask_filename).get_fdata() != 0
X = np.array([nibabel.load(img_filename).get_fdata()[mask_arr] for img_filename in data.img_filename])
assert X.shape == (84, 331695)
y = data.age.values

###############################################################################
# %% Split Data

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2)


###############################################################################
# %% Ridge regression


print("# Ridge regression ###################################################")

lrl2_cv = make_pipeline(
    preprocessing.StandardScaler(),
    GridSearchCV(lm.Ridge(),
                 {'alpha': 10. ** np.arange(-3, 3)},
                 cv=N_FOLDS_INNER, n_jobs=N_FOLDS_INNER))

lrl2_cv.fit(X=X_train, y=y_train)
y_pred_train = lrl2_cv.predict(X_train)
y_pred_test = lrl2_cv.predict(X_test)

print(cv_train_test_scores_params(lrl2_cv, y_pred_train, y_train, y_pred_test,
                                  y_test))

###############################################################################
# %% Elastic-net

print("# Elastic-net regressor #############################################")

enet_cv = make_pipeline(
    # preprocessing.MinMaxScaler(),
    preprocessing.StandardScaler(),
    GridSearchCV(estimator=lm.ElasticNet(max_iter=50000),
                 param_grid={'alpha': 10. ** np.arange(-3, 3),
                             'l1_ratio': [.1, .5, .9]},
                 cv=N_FOLDS_INNER, n_jobs=N_FOLDS_INNER))


enet_cv.fit(X=X_train, y=y_train)

y_pred_train = enet_cv.predict(X_train)
y_pred_test = enet_cv.predict(X_test)

print(cv_train_test_scores_params(enet_cv, y_pred_train, y_train, y_pred_test,
                                  y_test))

###############################################################################
# %% RandomForest regressor

print("# RandomForest regressor #############################################")

forest_cv = make_pipeline(
    preprocessing.MinMaxScaler(),
    GridSearchCV(estimator=RandomForestRegressor(random_state=1),
                 param_grid={"n_estimators": [10, 100],
                             'criterion': ['mse', 'mae']},
                 cv=N_FOLDS_INNER, n_jobs=N_FOLDS_INNER))

forest_cv.fit(X=X_train, y=y_train)
y_pred_train = forest_cv.predict(X_train)
y_pred_test = forest_cv.predict(X_test)

print(cv_train_test_scores_params(forest_cv, y_pred_train, y_train,
                                  y_pred_test, y_test))

###############################################################################
# %% GradientBoosting regressor

print("# GradientBoosting regressor #########################################")

gb_cv = make_pipeline(
    preprocessing.MinMaxScaler(),
    GridSearchCV(estimator=GradientBoostingRegressor(random_state=1),
                 param_grid={"n_estimators": [10, 100],
                             'loss': ['ls', 'lad', 'huber']},
                 cv=N_FOLDS_INNER, n_jobs=N_FOLDS_INNER))

gb_cv.fit(X=X_train, y=y_train)
y_pred_train = gb_cv.predict(X_train)
y_pred_test = gb_cv.predict(X_test)

print(cv_train_test_scores_params(gb_cv, y_pred_train, y_train, y_pred_test,
                                  y_test))

###############################################################################
# %% Multi-layer Perceptron (MLP) regressor

print("# Multi-layer Perceptron (MLP) regressor #############################")

param_grid = {"hidden_layer_sizes": [(100, ), (50, ), (25, ), (10, ), (5, ),          # 1 hidden layer
                                     (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),  # 2 hidden layers
                                     (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )], # 3 hidden layers
              "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}

mlp_cv = make_pipeline(
    preprocessing.MinMaxScaler(),
    GridSearchCV(estimator=MLPRegressor(random_state=1),
                 param_grid=param_grid,
                 cv=cv_train, n_jobs=N_FOLDS))

mlp_cv.fit(X=X_train, y=y_train)
y_pred_train = mlp_cv.predict(X_train)
y_pred_test = mlp_cv.predict(X_test)

print(cv_train_test_scores_params(mlp_cv, y_pred_train, y_train, y_pred_test,
                                  y_test))

###############################################################################
# %% Multiple models with model selection grid-search CV

# Build a dictionary of models

mlp_param_grid = {"hidden_layer_sizes":
                  [(100, ), (50, ), (25, ), (10, ), (5, ),          # 1 hidden layer
                   (100, 50, ), (50, 25, ), (25, 10, ), (10, 5, ),  # 2 hidden layers
                   (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )], # 3 hidden layers
                  "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}

models = dict(
    lrl2_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(lm.Ridge(),
                     param_grid={'alpha': 10. ** np.arange(-1, 3)},
                     cv=cv_train, n_jobs=N_FOLDS)),

    lrenet_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(lm.ElasticNet(max_iter=1000),
                     param_grid={'alpha': 10. ** np.arange(-1, 2),
                                 'l1_ratio': [.1, .5]},
                     cv=cv_train, n_jobs=N_FOLDS)),

    svmrbf_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(svm.SVR(),
                     # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                     param_grid={'kernel': ['poly', 'rbf'],
                                 'C': 10. ** np.arange(-1, 2)},
                     cv=cv_train, n_jobs=N_FOLDS)),

    forest_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(RandomForestRegressor(random_state=1),
                     param_grid={"n_estimators": [100]},
                     cv=cv_train, n_jobs=N_FOLDS)),

    gb_cv=make_pipeline(
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=GradientBoostingRegressor(random_state=1),
                     param_grid={"n_estimators": [100],
                                 "subsample":[1, .5],
                                 "learning_rate": [.1, .5]
                                 },
                     cv=cv_train, n_jobs=N_FOLDS)),

    mlp_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=MLPRegressor(random_state=1),
                     param_grid=param_grid,
                     cv=cv_train, n_jobs=N_FOLDS)))


###############################################################################
# %% Run models
# Run on training set:
# - Iterate over models
# - Grid search CV model selection on training data

# X_train = vols_train
# X_test  = vols_test

perfs_cv_rois = pd.DataFrame()

for name, model in models.items():
    # name, model = "lrenet_cv", models["lrenet_cv"]
    # name, model = "svmrbf_cv", models["svmrbf_cv"]
    # name, model = "forest_cv", models["forest_cv"]

    start_time = time.time()
    model.fit(X=X_train, y=y_train)
    print(name, 'elapsed time: \t%.3f sec' % (time.time() - start_time))
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    perfs_ = cv_train_test_scores_params(model, y_pred_train, y_train,
                                         y_pred_test, y_test)
    perfs_.insert(0, "model", name)
    perfs_cv_rois = perfs_cv_rois.append(perfs_)

# %%
