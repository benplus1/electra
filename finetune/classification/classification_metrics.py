# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation metrics for classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import scipy
import sklearn
from util import utils


from finetune import scorer


class SentenceLevelScorer(scorer.Scorer):
  """Abstract scorer for classification/regression tasks."""

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(SentenceLevelScorer, self).__init__()
    self._total_loss = 0
    self._true_labels = []
    self._preds = []
    self._cls_ids = []
    self._probabilities = []
    self._logits = []

  def update(self, results):
    super(SentenceLevelScorer, self).update(results)
    utils.log(results)
    self._total_loss += results['loss']
    self._true_labels.append(results['label_ids'])
    self._preds.append(results['predictions'])
    self._cls_ids.append(results['cls_ids'])
    self._probabilities.append(results['probabilities'])
    self._logits.append(results['logits'])
    # self._cls_ids.append(results['cls_ids'])

  def get_loss(self):
    return self._total_loss / len(self._true_labels)


class AccuracyScorer(SentenceLevelScorer):

  def _get_results(self):
    correct, count = 0, 0
    for y_true, pred in zip(self._true_labels, self._preds):
      count += 1
      correct += (1 if y_true == pred else 0)
    return [
        ('accuracy', 100.0 * correct / count),
        ('loss', self.get_loss()),
    ]


class F1Scorer(SentenceLevelScorer):
  """Computes F1 for classification tasks."""

  def __init__(self):
    super(F1Scorer, self).__init__()
    self._positive_label = 1

  def _get_results(self):
    n_correct, n_predicted, n_gold = 0, 0, 0
    for y_trues, preds, cls_ids in zip(self._true_labels, self._preds, self._cls_ids):
      utils.log("HERE")
      utils.log(y_trues)
      utils.log(preds)
      utils.log(cls_ids)
      for i in self._probabilities:
        utils.log(i)
      for y_true, pred, cls_id in zip(y_trues, preds, cls_ids):
        if cls_id == 1: # cls_id good
          if pred == self._positive_label:
            n_predicted += 1
          if y_true == self._positive_label:
            n_gold += 1
          if y_true == pred and pred == self._positive_label:
            n_correct += 1
    if n_correct == 0:
      p, r, f1 = 0, 0, 0
    else:
      p = n_correct / n_predicted
      r = n_correct / n_gold
      f1 = 2 * p * r / (p + r)
    return [
        ('precision', p),
        ('recall', r),
        ('f1', f1),
        ('loss', self.get_loss()),
        ('labels', self._true_labels),
        ('probabilities', self._probabilities),
        ('preds', self._preds),
        ('cls_ids', self._cls_ids),
        ('logits', self._logits),
    ]


class MCCScorer(SentenceLevelScorer):

  def _get_results(self):
    return [
        ('mcc', 100 * sklearn.metrics.matthews_corrcoef(
            self._true_labels, self._preds)),
        ('loss', self.get_loss()),
    ]


class RegressionScorer(SentenceLevelScorer):

  def _get_results(self):
    preds = np.array(self._preds).flatten()
    return [
        ('pearson', 100.0 * scipy.stats.pearsonr(
            self._true_labels, preds)[0]),
        ('spearman', 100.0 * scipy.stats.spearmanr(
            self._true_labels, preds)[0]),
        ('mse', np.mean(np.square(np.array(self._true_labels) - self._preds))),
        ('loss', self.get_loss()),
    ]
