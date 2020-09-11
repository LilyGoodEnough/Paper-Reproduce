#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:06:40 2020

@author: mac
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time                        

from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


# data process
canceData=load_breast_cancer()
X=canceData.data
y=canceData.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# model definition    
def run_xgb_exa(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()           
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    xgb_exa = XGBClassifier( learning_rate =0.1, n_estimators=1000,
                         max_depth=5, min_child_weight=1, gamma=0,
                         subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                         nthread=2, scale_pos_weight=1, seed=27, tree_method = 'exact')
    modelfit(xgb_exa, train, predictors)
    end_time = time.time()             
    run_time = end_time - start_time
    acc = metrics.accuracy_score(y_test,y_pre)
    auc = metrics.roc_auc_score(y_test,y_pre)
    return acc, auc, run_time

  
def run_xgb_his(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()           
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    xgb_his = XGBClassifier( learning_rate =0.1, n_estimators=1000,
                         max_depth=5, min_child_weight=1, gamma=0,
                         subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                         nthread=2, scale_pos_weight=1, seed=27, tree_method = 'hist')
    modelfit(xgb_exa, train, predictors)
    end_time = time.time()             
    run_time = end_time - start_time
    acc = metrics.accuracy_score(y_test,y_pre)
    auc = metrics.roc_auc_score(y_test,y_pre)
    return acc, auc, run_time

  
def run_lgb_baseline(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()           
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    model = GradientBoostingClassifier(boosting_type='gbdt', objective='binary', metrics='auc',
                             learning_rate=0.01, n_estimators=1000, max_depth=4, 
                             num_leaves=10, max_bin=255, min_data_in_leaf=81,
                             bagging_fraction=0.7, bagging_freq= 30, 
                             feature_fraction= 0.8, lambda_l1=0.1,lambda_l2=0,
                             min_split_gain=0.1)
    model.fit(X_train,y_train)
    y_pre = model.predict(X_test)
    end_time = time.time()             
    run_time = end_time - start_time
    acc = metrics.accuracy_score(y_test,y_pre)
    auc = metrics.roc_auc_score(y_test,y_pre)
    return acc, auc, run_time


def run_lgb_SGB(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()           
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    model = GradientBoostingClassifier( boosting_type='gbdt', objective='binary', metrics='auc',
                             learning_rate=0.01, n_estimators=1000, max_depth=4, 
                             num_leaves=10, max_bin=255, min_data_in_leaf=81,
                             bagging_fraction=0.7, bagging_freq= 30, 
                             feature_fraction= 0.8, lambda_l1=0.1,lambda_l2=0,
                             min_split_gain=0.1)
    model.fit(X_train,y_train)
    y_pre = model.predict(X_test)
    end_time = time.time()             
    run_time = end_time - start_time
    acc = metrics.accuracy_score(y_test,y_pre)
    auc = metrics.roc_auc_score(y_test,y_pre)
    return acc, auc, run_time

 
def run_lightGBM(X_train, y_train, X_test, y_test, get_training=False):
    start_time = time.time()           
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    model=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc',
                             learning_rate=0.01, n_estimators=1000, max_depth=4, 
                             num_leaves=10, max_bin=255, min_data_in_leaf=81,
                             bagging_fraction=0.7, bagging_freq= 30, 
                             feature_fraction= 0.8, lambda_l1=0.1,lambda_l2=0,
                             min_split_gain=0.1)
    model.fit(X_train,y_train)
    y_pre = model.predict(X_test)
    end_time = time.time()             
    run_time = end_time - start_time
    acc = metrics.accuracy_score(y_test,y_pre)
    auc = metrics.roc_auc_score(y_test,y_pre)
    return acc, auc, run_time

"""
# plot graph       
def plot_precision_curve(extra_plot_title, 
                         model_name="Model",
                         colors=["blue", "darkorange", "brown", "red", "purple"],
                         legend_loc=None,figure_size=None,ylim=None):
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    title = "Precision Curve" if extra_plot_title == "" else extra_plot_title
    plt.title(title, fontsize=20)
    colors = colors + list(cm.rainbow(np.linspace(0, 1, len(final_TPs))))
    
    plt.xlabel("Time(s)", fontsize=18)
    plt.ylabel("AUC", fontsize=18)
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


# run general
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

  for i in range(len(all_signal_names)):
    final_TPs[i] = np.array(final_TPs[i])
    final_stderrs[i] = np.array(final_stderrs[i])

  plot_precision_curve(extra_plot_title, percentile_levels, all_signal_names,
                       final_TPs, final_stderrs, final_misclassification)
  return (all_signal_names, final_TPs, final_stderrs)
"""