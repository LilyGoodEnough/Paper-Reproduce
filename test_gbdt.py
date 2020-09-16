#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:06:40 2020

@author: lily
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time                        

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.cm as cm
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

# data process
canceData=load_breast_cancer()
X=canceData.data
y=canceData.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)



# Trust Score Class -- see paper NIPS18 (To Trust or Not to Trust a Classifier)
class TrustScore:
  """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
  """

  def __init__(self, k=10, alpha=0., filtering="none", min_dist=1e-12):
    """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
    self.k = k
    self.filtering = filtering
    self.alpha = alpha
    self.min_dist = min_dist

  def filter_by_density(self, X):
    """Filter out points with low kNN density.

    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    original points of kNN density.
    """
    kdtree = KDTree(X)
    knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
    eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
    return X[np.where(knn_radii <= eps)[0], :]

  def filter_by_uncertainty(self, X, y):
    """Filter out points with high label disagreement amongst its kNN neighbors.

    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    samples with highest disagreement amongst its k nearest neighbors.
    """
    neigh = KNeighborsClassifier(n_neighbors=self.k)
    neigh.fit(X, y)
    confidence = neigh.predict_proba(X)
    cutoff = np.percentile(confidence, self.alpha * 100)
    unfiltered_idxs = np.where(confidence >= cutoff)[0]
    return X[unfiltered_idxs, :], y[unfiltered_idxs]

  def fit(self, X, y):
    """Initialize trust score precomputations with training data.

    WARNING: assumes that the labels are 0-indexed (i.e.
    0, 1,..., n_labels-1).

    Args:
    X: an array of sample points.
    y: corresponding labels.
    """
    self.n_labels = np.max(y) + 1
    self.kdtrees = [None] * self.n_labels
    if self.filtering == "uncertainty":
      X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
    for label in range(self.n_labels):
      if self.filtering == "none":
        X_to_use = X[np.where(y == label)[0]]
        self.kdtrees[label] = KDTree(X_to_use)
      elif self.filtering == "density":
        X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
        self.kdtrees[label] = KDTree(X_to_use)
      elif self.filtering == "uncertainty":
        X_to_use = X_filtered[np.where(y_filtered == label)[0]]
        self.kdtrees[label] = KDTree(X_to_use)

      if len(X_to_use) == 0:
        print(
            "Filtered too much or missing examples from a label! Please lower "
            "alpha or check data.")

  def get_score(self, X, y_pred):
    """Compute the trust scores.

    Given a set of points, determines the distance to each class.

    Args:
    X: an array of sample points.
    y_pred: The predicted labels for these points.

    Returns:
    The trust score, which is ratio of distance to closest class that was not
    the predicted class to the distance to the predicted class.
    """
    d = np.tile(None, (X.shape[0], self.n_labels))
    for label_idx in range(self.n_labels):
      d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]

    sorted_d = np.sort(d, axis=1)
    d_to_pred = d[range(d.shape[0]), y_pred]
    d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                     sorted_d[:, 0], sorted_d[:, 1])
    return d_to_closest_not_pred / (d_to_pred + self.min_dist)


class KNNConfidence:
  """Baseline which uses disagreement to kNN classifier.
  """

  def __init__(self, k=10):
    self.k = k

  def fit(self, X, y):
    self.kdtree = KDTree(X)
    self.y = y

  def get_score(self, X, y_pred):
    knn_idxs = self.kdtree.query(X, k=self.k)[1]
    knn_outputs = self.y[knn_idxs]
    return np.mean(
        knn_outputs == np.transpose(np.tile(y_pred, (self.k, 1))), axis=1)




# model definition    
def run_xgb_exa(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()
    model = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000,
                         max_depth=5, min_child_weight=1, gamma=0,
                         subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                         nthread=2, scale_pos_weight=1, seed=27, tree_method = 'exact')          
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    #if not get_training:
    #    return y_pred, confidences
    end_time = time.time()             
    run_time = end_time - start_time
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
    acc = metrics.accuracy_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred)    
    return acc, auc, run_time

  
def run_xgb_his(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()
    model = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000,
                         max_depth=5, min_child_weight=1, gamma=0,
                         subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                         nthread=2, scale_pos_weight=1, seed=27, tree_method = 'hist')          
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    #if not get_training:
    #    return y_pred, confidences
    end_time = time.time()             
    run_time = end_time - start_time
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
    acc = metrics.accuracy_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred)    
    return acc, auc, run_time
  

def run_lgb_baseline(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()
    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000, max_depth=4, 
                                       min_samples_leaf = 10, min_samples_split = 81, 
                                       max_features=9, subsample=0.7, random_state = None)          
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    #if not get_training:
    #    return y_pred, confidences
    end_time = time.time()             
    run_time = end_time - start_time
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
    acc = metrics.accuracy_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred)    
    return acc, auc, run_time


def run_lgb_SGB(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()
    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000, max_depth=4, 
                                       min_samples_leaf = 10, min_samples_split = 81, 
                                       max_features=9, subsample=0.7, random_state=15 )          
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    #if not get_training:
    #    return y_pred, confidences
    end_time = time.time()             
    run_time = end_time - start_time
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
    acc = metrics.accuracy_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred)    
    return acc, auc, run_time 


def run_lightGBM(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()           
    model=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc',
                             learning_rate=0.1, n_estimators=1000, max_depth=4, 
                             num_leaves=10, max_bin=255, min_child_samples=81,
                             subsample=0.7, subsample_freq=30, 
                             colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0,
                             min_split_gain=0.1)       
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    all_confidence = model.predict_proba(X_test)
    confidences = all_confidence[range(len(y_pred)), y_pred]
    if not get_training:
        return y_pred, confidences
    end_time = time.time()             
    run_time = end_time - start_time
    y_pred_training = model.predict(X_train)
    all_confidence_training = model.predict_proba(X_train)
    confidence_training = all_confidence_training[range(len(y_pred_training)),
                                                y_pred_training]
    acc = metrics.accuracy_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred)    
    return acc, auc, run_time


# plot graph       
def plot_precision_curve(
    extra_plot_title,
    percentile_levels,
    signal_names,
    final_TPs,
    final_stderrs,
    final_misclassification,
    model_name="Model",
    colors=["blue", "darkorange", "brown", "red", "purple"],
    legend_loc=None,
    figure_size=None,
    ylim=None):
  if figure_size is not None:
    plt.figure(figsize=figure_size)
  title = "Precision Curve" if extra_plot_title == "" else extra_plot_title
  plt.title(title, fontsize=20)
  colors = colors + list(cm.rainbow(np.linspace(0, 1, len(final_TPs))))

  plt.xlabel("Percentile level", fontsize=18)
  plt.ylabel("Precision", fontsize=18)
  for i, signal_name in enumerate(signal_names):
    ls = "--" if ("Model" in signal_name) else "-"
    plt.plot(
        percentile_levels, final_TPs[i], ls, c=colors[i], label=signal_name)

    plt.fill_between(
        percentile_levels,
        final_TPs[i] - final_stderrs[i],
        final_TPs[i] + final_stderrs[i],
        color=colors[i],
        alpha=0.1)

  if legend_loc is None:
    if 0. in percentile_levels:
      plt.legend(loc="lower right", fontsize=14)
    else:
      plt.legend(loc="upper left", fontsize=14)
  else:
    if legend_loc == "outside":
      plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=14)
    else:
      plt.legend(loc=legend_loc, fontsize=14)
  if ylim is not None:
    plt.ylim(*ylim)
  model_acc = 100 * (1 - final_misclassification)
  plt.axvline(x=model_acc, linestyle="dotted", color="black")
  plt.show()


def run_precision_recall_experiment_general(X,
                                            y,
                                            n_repeats,
                                            percentile_levels,
                                            trainer,
                                            test_size=0.5,
                                            extra_plot_title="",
                                            signals=[],
                                            signal_names=[],
                                            predict_when_correct=False,
                                            skip_print=False):

  def get_stderr(L):
    return np.std(L) / np.sqrt(len(L))

  all_signal_names = ["Model Confidence"] + signal_names
  all_TPs = [[[] for p in percentile_levels] for signal in all_signal_names]
  misclassifications = []
  sign = 1 if predict_when_correct else -1
  sss = StratifiedShuffleSplit(
      n_splits=n_repeats, test_size=test_size, random_state=0)
  for train_idx, test_idx in sss.split(X, y):
    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]
    testing_prediction, testing_confidence_raw = trainer(
        X_train, y_train, X_test, y_test)
    target_points = np.where(
        testing_prediction == y_test)[0] if predict_when_correct else np.where(
            testing_prediction != y_test)[0]

    final_signals = [testing_confidence_raw]
    for signal in signals:
      signal.fit(X_train, y_train)
      final_signals.append(signal.get_score(X_test, testing_prediction))

    for p, percentile_level in enumerate(percentile_levels):
      all_high_confidence_points = [
          np.where(sign * signal >= np.percentile(sign *
                                                  signal, percentile_level))[0]
          for signal in final_signals
      ]

      if 0 in map(len, all_high_confidence_points):
        continue
      TP = [
          len(np.intersect1d(high_confidence_points, target_points)) /
          (1. * len(high_confidence_points))
          for high_confidence_points in all_high_confidence_points
      ]
      for i in range(len(all_signal_names)):
        all_TPs[i][p].append(TP[i])
    misclassifications.append(len(target_points) / (1. * len(X_test)))

  final_TPs = [[] for signal in all_signal_names]
  final_stderrs = [[] for signal in all_signal_names]
  for p, percentile_level in enumerate(percentile_levels):
    for i in range(len(all_signal_names)):
      final_TPs[i].append(np.mean(all_TPs[i][p]))
      final_stderrs[i].append(get_stderr(all_TPs[i][p]))

    if not skip_print:
      print("Precision at percentile", percentile_level)
      ss = ""
      for i, signal_name in enumerate(all_signal_names):
        ss += (signal_name + (": %.4f  " % final_TPs[i][p]))
      print(ss)
      print()

  final_misclassification = np.mean(misclassifications)

  if not skip_print:
    print("Misclassification rate mean/std", np.mean(misclassifications),
          get_stderr(misclassifications))

  for i in range(len(all_signal_names)):
    final_TPs[i] = np.array(final_TPs[i])
    final_stderrs[i] = np.array(final_stderrs[i])

  plot_precision_curve(extra_plot_title, percentile_levels, all_signal_names,
                       final_TPs, final_stderrs, final_misclassification)
  return (all_signal_names, final_TPs, final_stderrs, final_misclassification)


# main function
xgb_exa = run_xgb_exa(X_train, y_train, X_test, y_test, get_training=False)
xgb_his = run_xgb_his(X_train, y_train, X_test, y_test, get_training=False)
lgb_base = run_lgb_baseline(X_train, y_train, X_test, y_test, get_training=False)
lgb_sgb = run_lgb_SGB(X_train, y_train, X_test, y_test, get_training=False)
lgbm = run_lightGBM(X_train, y_train, X_test, y_test, get_training=False)
print(xgb_exa,'\n',xgb_his,'\n',lgb_base,'\n',lgb_sgb,'\n',lgbm)

kconfidence = KNNConfidence()
trustscore = TrustScore()
auc = run_precision_recall_experiment_general(X,
                                            y,
                                            20,
                                            [10,20,30,40,50,60,70,80,90,100],
                                            run_lightGBM,
                                            test_size=0.5,
                                            extra_plot_title="LightGBM Precision Curve",
                                            signals=[trustscore,kconfidence],
                                            signal_names=["Trust Score","KNN Confidence"],
                                            predict_when_correct=False,
                                            skip_print=False)



print(auc)
