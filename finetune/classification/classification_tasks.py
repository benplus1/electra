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

"""Text classification and regression tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import csv
import os
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from finetune.classification import classification_metrics
from model import tokenization
from util import utils
import copy

class InputExample(task.Example):
  """A single training/test example for simple sequence classification."""

  def __init__(self, eid, task_name, text_a, labels=None, cls_locs=None):
    super(InputExample, self).__init__(task_name)
    self.eid = eid
    self.text_a = text_a
    self.labels = labels # holds the correct labels according to the CLS ids
    self.cls_locs = cls_locs

def read_txt(input_file, quotechar=None):
  """Reads a tab separated value file."""
  with tf.io.gfile.GFile(input_file, "r") as f:
    # reader = csv.reader(f, delimiter=" ", quotechar=quotechar)
    lines = []
    for line in f:
      lines.append(line)
    return lines

class SingleOutputTask(task.Task):
  """Task with a single prediction per example (e.g., text classification)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(SingleOutputTask, self).__init__(config, name)
    self._tokenizer = tokenizer

  def get_examples(self, split):
    # return self._create_examples(read_tsv(
    #     os.path.join(self.config.raw_data_dir(self.name), split + ".tsv"),
    #     max_lines=100 if self.config.debug else None), split)
    return self._create_examples(read_txt(os.path.join(self.config.raw_data_dir(self.name), split +".txt")), split)

  @abc.abstractmethod
  def _create_examples(self, lines, split):
    pass

  def featurize(self, example: InputExample, is_training, log=False):
    """Turn an InputExample into a dict of features."""
    tokens_a = self._tokenizer.tokenize(example.text_a)

    # tokens_b = None
    # if example.text_b:
    #   tokens_b = self._tokenizer.tokenize(example.text_b)

    # if tokens_b:
    #   # Modifies `tokens_a` and `tokens_b` in place so that the total
    #   # length is less than the specified length.
    #   # Account for [CLS], [SEP], [SEP] with "- 3"
    #   _truncate_seq_pair(tokens_a, tokens_b, self.config.max_seq_length - 3)
    # else:
    #   # Account for [CLS] and [SEP] with "- 2"
    #   if len(tokens_a) > self.config.max_seq_length - 3: # change from 2 to 3
    #     tokens_a = tokens_a[0:(self.config.max_seq_length - 3)] # change from 2 to 3

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it
    # makes it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    ex_cls_locs = example.cls_locs

    tokens = []
    segment_ids = []
    cls_ids = [] # new

    for (i, token) in enumerate(tokens_a):
      if i in ex_cls_locs:
        tokens.append("[CLS]")
        segment_ids.append(0)
        cls_ids.append(len(tokens)-1)
      tokens.append(token)
      segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < self.config.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
    
    cls_id_end = len(cls_ids)

    while len(cls_ids) < self.config.max_seq_length:
      cls_ids.append(-2)

    if log:
      utils.log("  Example {:}".format(example.eid))
      utils.log("    tokens: {:}".format(" ".join(
          [tokenization.printable_text(x) for x in tokens])))
      utils.log("    input_ids: {:}".format(" ".join(map(str, input_ids))))
      utils.log("    input_mask: {:}".format(" ".join(map(str, input_mask))))
      utils.log("    segment_ids: {:}".format(" ".join(map(str, segment_ids))))
      utils.log("    cls_ids: {:}".format(" ".join(map(str, cls_ids))))
      utils.log(self.config.max_seq_length)
    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length

    eid = example.eid
    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_cls_ids": cls_ids,
        self.name + "_cls_id_end": cls_id_end,
        self.name + "_eid": eid,
    }
    self._add_features(features, example, log)
    return features

  def _load_glue(self, lines, split, is_state_tok, is_pos_tok, cor_tok, neg_tok):
    examples = []
    text_a_buf = ""
    curr_labels = []
    curr_cls_locs = []
    curr_eid_i = 0
    for (i, line) in enumerate(lines):
      try:
        if (i % 4 == 0): # it's the text itself
          if (line == '\n'): # its the end of the file
            return examples
          else:
            text_a_buf = tokenization.convert_to_unicode(line)
        elif (i % 4 == 1): # its the start cls
          for (j, k) in enumerate(line.split()):
            if k == is_state_tok:
              curr_cls_locs.append(j)
        elif (i % 4 == 2): # its the labels
          for k in line.split():
            actual_val = cor_tok if k == is_pos_tok else neg_tok
            label = tokenization.convert_to_unicode(actual_val)
            curr_labels.append(label)
        elif (i % 4 == 3): # its the new line. 
          if len(text_a_buf < 500):
            examples.append(InputExample(eid=curr_eid_i, task_name=self.name,
                                        text_a=copy.deepcopy(text_a_buf), labels=copy.deepcopy(curr_labels), cls_locs=copy.deepcopy(curr_cls_locs)))
          # clean buffers
          text_a_buf = ""
          curr_labels = []
          curr_cls_locs = []
          curr_eid_i += 1
      except Exception as ex:
        utils.log("Error constructing example from line", i,
                  "for task", self.name + ":", ex)
        utils.log("Input causing the error:", line)
    return examples

  @abc.abstractmethod
  def _get_dummy_label(self):
    pass

  @abc.abstractmethod
  def _add_features(self, features, example, log):
    pass

class ClassificationTask(SingleOutputTask):
  """Task where the output is a single categorical label for the input text."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, label_list):
    super(ClassificationTask, self).__init__(config, name, tokenizer)
    self._tokenizer = tokenizer
    self._label_list = label_list # always [0, 1]?

  # need to change this for test set data set
  def _get_dummy_label(self):
    return self._label_list[0]

  def get_feature_specs(self):
    return [feature_spec.FeatureSpec(self.name + "_eid", []),
            feature_spec.FeatureSpec(self.name + "_label_ids", [self.config.max_seq_length]),
            feature_spec.FeatureSpec(self.name + "_cls_ids", [self.config.max_seq_length]),
            feature_spec.FeatureSpec(self.name + "_cls_id_end", [])]

  def _add_features(self, features, example, log):
    label_map = {}
    for (i, label) in enumerate(self._label_list):
      label_map[label] = i
    label_ids = []
    for label in example.labels:
      label_ids.append(label_map[label])
    if log:
      utils.log("    labels: {:} (id = {:})".format(example.labels, label_ids))
    features[example.task_name + "_label_ids"] = label_ids

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    utils.log('here')
    num_labels = len(self._label_list)
    reprs = bert_model.get_sequence_output() # a list of all seq_length

    utils.log(reprs)
    label_ids =  features[self.name + "_label_ids"]

    if is_training:
      reprs = tf.nn.dropout(reprs, keep_prob=0.9) # dropout looks at everything, so this is fine

    # cls_ids is now fixed, -2 as padding.
    # get index of max number:
    all_cls = tf.squeeze(features['sst_cls_ids'])
    utils.log(all_cls)
    index_cor = tf.squeeze(features['sst_cls_id_end'])
    correct_cls = all_cls[:index_cor]
    utils.log(correct_cls)
    reprs = tf.gather(reprs, correct_cls, axis=1)
    utils.log(features)
    utils.log(reprs)
    # sequence_output: [batch_size, seq_length, hidden_size]
    # pooled_output: [batch_size, hidden_size]
    # layers_dense goes from [batch_size, hidden_size] -> [batch_size, 2] (last dimension becomes 2)
    # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, 2]
    logits = tf.layers.dense(reprs, num_labels) # reprs is supposed to be pooledoutput, but is currently sequenceoutput
    # same shape as logits -> [batch_size, 2]
    # [batch_size, seq_length, 2]
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # usually, label_id is a scalar, which returns an output_shape of vector length depth (2)
    # now, indices is a vector of length features (num_cls_ids), which retursn features x depth for axis == -1, depth x features for axis == 0
    # [seq_length, 2] or [2, seq_length]
    labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32, axis=-1)

    # labels is vector of length 2, log_probs is tensor of shape [batch_size, 2]
    # losses = -tf.reduce_sum(labels * log_probs, axis=-1) -> [batch_size, ]
    losses = -tf.reduce_sum(labels * log_probs)

    # logits -> [batch_size, 2] -> [batch_size, ]
    # logits -> [batch_size, seq_length, 2] -> [batch_size, seq_length]
    redictions = tf.argmax(logits, axis=-1)

    robabilities = tf.nn.softmax(logits)

    outputs = dict(
        pool_output=reprs,
        loss=losses,
        logits=logits,
        predictions=redictions,
        probabilities=robabilities,
        label_ids=label_ids,
        eid=features[self.name + "_eid"],
    )
    return losses, outputs

  def get_scorer(self):
    return classification_metrics.F1Scorer()


class SST(ClassificationTask):
  """Stanford Sentiment Treebank."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(SST, self).__init__(config, "sst", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    if "test" in split:
      return self._load_glue(lines, split, 'S', 'P', '1', '0')
    else:
      return self._load_glue(lines, split, 'S', 'P', '1', '0')


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#   """Truncates a sequence pair in place to the maximum length."""

#   # This is a simple heuristic which will always truncate the longer sequence
#   # one token at a time. This makes more sense than truncating an equal percent
#   # of tokens from each, since if one sequence is very short then each token
#   # that's truncated likely contains more information than a longer sequence.
#   while True:
#     total_length = len(tokens_a) + len(tokens_b)
#     if total_length <= max_length:
#       break
#     if len(tokens_a) > len(tokens_b):
#       tokens_a.pop()
#     else:
#       tokens_b.pop()


  # def _load_glue(self, lines, split, text_a_loc, text_b_loc, label_loc,
  #                skip_first_line=False, eid_offset=0, swap=False):
  #   examples = []
  #   for (i, line) in enumerate(lines):
  #     try:
  #       if i == 0 and skip_first_line:
  #         continue
  #       eid = i - (1 if skip_first_line else 0) + eid_offset
  #       text_a = tokenization.convert_to_unicode(line[text_a_loc])
  #       if text_b_loc is None:
  #         text_b = None
  #       else:
  #         text_b = tokenization.convert_to_unicode(line[text_b_loc])
  #       if "test" in split or "diagnostic" in split:
  #         label = self._get_dummy_label()
  #       else:
  #         label = tokenization.convert_to_unicode(line[label_loc])
  #       if swap:
  #         text_a, text_b = text_b, text_a
  #       examples.append(InputExample(eid=eid, task_name=self.name,
  #                                    text_a=text_a, text_b=text_b, label=label)) ########## change to array
  #     except Exception as ex:
  #       utils.log("Error constructing example from line", i,
  #                 "for task", self.name + ":", ex)
  #       utils.log("Input causing the error:", line)
  #   return examples

# def read_tsv(input_file, quotechar=None, max_lines=None):
#   """Reads a tab separated value file."""
#   with tf.io.gfile.GFile(input_file, "r") as f:
#     reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#     lines = []
#     for i, line in enumerate(reader):
#       if max_lines and i >= max_lines:
#         break
#       lines.append(line)
#     return lines

# class RegressionTask(SingleOutputTask):
#   """Task where the output is a real-valued score for the input text."""

#   __metaclass__ = abc.ABCMeta

#   def __init__(self, config: configure_finetuning.FinetuningConfig, name,
#                tokenizer, min_value, max_value):
#     super(RegressionTask, self).__init__(config, name, tokenizer)
#     self._tokenizer = tokenizer
#     self._min_value = min_value
#     self._max_value = max_value

#   def _get_dummy_label(self):
#     return 0.0

#   def get_feature_specs(self):
#     feature_specs = [feature_spec.FeatureSpec(self.name + "_eid", []),
#                      feature_spec.FeatureSpec(self.name + "_targets", [],
#                                               is_int_feature=False)]
#     return feature_specs

#   def _add_features(self, features, example, log):
#     label = float(example.label)
#     assert self._min_value <= label <= self._max_value
#     # simple normalization of the label
#     label = (label - self._min_value) / self._max_value
#     if log:
#       utils.log("    label: {:}".format(label))
#     features[example.task_name + "_targets"] = label

#   def get_prediction_module(self, bert_model, features, is_training,
#                             percent_done):
#     reprs = bert_model.get_pooled_output()
#     if is_training:
#       reprs = tf.nn.dropout(reprs, keep_prob=0.9)

#     predictions = tf.layers.dense(reprs, 1)
#     predictions = tf.squeeze(predictions, -1)

#     targets = features[self.name + "_targets"]
#     losses = tf.square(predictions - targets)
#     outputs = dict(
#         loss=losses,
#         predictions=predictions,
#         targets=features[self.name + "_targets"],
#         eid=features[self.name + "_eid"]
#     )
#     return losses, outputs

#   def get_scorer(self):
#     return classification_metrics.RegressionScorer()

# class MNLI(ClassificationTask):
#   """Multi-NLI."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(MNLI, self).__init__(config, "mnli", tokenizer,
#                                ["contradiction", "entailment", "neutral"])

#   def get_examples(self, split):
#     if split == "dev":
#       split += "_matched"
#     return self._create_examples(read_tsv(
#         os.path.join(self.config.raw_data_dir(self.name), split + ".tsv"),
#         max_lines=100 if self.config.debug else None), split)

#   def _create_examples(self, lines, split):
#     if split == "diagnostic":
#       return self._load_glue(lines, split, 1, 2, None, True)
#     else:
#       return self._load_glue(lines, split, 8, 9, -1, True)

#   def get_test_splits(self):
#     return ["test_matched", "test_mismatched", "diagnostic"]


# class MRPC(ClassificationTask):
#   """Microsoft Research Paraphrase Corpus."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(MRPC, self).__init__(config, "mrpc", tokenizer, ["0", "1"])

#   def _create_examples(self, lines, split):
#     examples = []
#     examples += self._load_glue(lines, split, 3, 4, 0, True)
#     if self.config.double_unordered and split == "train":
#       examples += self._load_glue(
#           lines, split, 3, 4, 0, True, len(examples), True)
#     return examples


# class CoLA(ClassificationTask):
#   """Corpus of Linguistic Acceptability."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(CoLA, self).__init__(config, "cola", tokenizer, ["0", "1"])

#   def _create_examples(self, lines, split):
#     return self._load_glue(lines, split, 1 if split == "test" else 3,
#                            None, 1, split == "test")

#   def get_scorer(self):
#     return classification_metrics.MCCScorer()


# class QQP(ClassificationTask):
#   """Quora Question Pair."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(QQP, self).__init__(config, "qqp", tokenizer, ["0", "1"])

#   def _create_examples(self, lines, split):
#     return self._load_glue(lines, split, 1 if split == "test" else 3,
#                            2 if split == "test" else 4, 5, True)


# class RTE(ClassificationTask):
#   """Recognizing Textual Entailment."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(RTE, self).__init__(config, "rte", tokenizer,
#                               ["entailment", "not_entailment"])

#   def _create_examples(self, lines, split):
#     return self._load_glue(lines, split, 1, 2, 3, True)


# class QNLI(ClassificationTask):
#   """Question NLI."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(QNLI, self).__init__(config, "qnli", tokenizer,
#                                ["entailment", "not_entailment"])

#   def _create_examples(self, lines, split):
#     return self._load_glue(lines, split, 1, 2, 3, True)


# class STS(RegressionTask):
#   """Semantic Textual Similarity."""

#   def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
#     super(STS, self).__init__(config, "sts", tokenizer, 0.0, 5.0)

#   def _create_examples(self, lines, split):
#     examples = []
#     if split == "test":
#       examples += self._load_glue(lines, split, -2, -1, None, True)
#     else:
#       examples += self._load_glue(lines, split, -3, -2, -1, True)
#     if self.config.double_unordered and split == "train":
#       examples += self._load_glue(
#           lines, split, -3, -2, -1, True, len(examples), True)
#     return examples
