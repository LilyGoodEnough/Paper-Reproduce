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

import matplotlib.cm as cm
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

# data process
canceData=load_breast_cancer()
X=canceData.data
y=canceData.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


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

xgb_exa = run_xgb_exa(X_train, y_train, X_test, y_test, get_training=False)
print(xgb_exa)
  
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

xgb_his = run_xgb_his(X_train, y_train, X_test, y_test, get_training=False)
print(xgb_his)  

def run_lgb_baseline(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()
    model = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1000, max_depth=4, 
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

lgb_base = run_lgb_baseline(X_train, y_train, X_test, y_test, get_training=False)
print(lgb_base) 


def run_lgb_SGB(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()
    model = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1000, max_depth=4, 
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

lgb_sgb = run_lgb_SGB(X_train, y_train, X_test, y_test, get_training=False)
print(lgb_sgb) 


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

lgbm = run_lightGBM(X_train, y_train, X_test, y_test, get_training=False)
print(lgbm)


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


auc = run_precision_recall_experiment_general(X,
                                            y,
                                            20,
                                            [10,20,30,40,50,60,70,80,90,100],
                                            run_xgb_his,
                                            test_size=0.5,
                                            extra_plot_title="",
                                            signals=[],
                                            signal_names=[],
                                            predict_when_correct=False,
                                            skip_print=False)

print(auc)
