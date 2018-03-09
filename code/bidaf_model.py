# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BiAttn
from qa_model import QAModel

logging.basicConfig(level=logging.INFO)

class SingleBiDAFModel(QAModel):
    """BiDAF Question Answering module"""

    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        with vs.variable_scope("Encoder"):
        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BiAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, U_tilde, _, H_tilde = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens, self.context_mask) # shapes are each (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, U_tilde, context_hiddens * U_tilde, context_hiddens * H_tilde], axis=2) # (batch_size, context_len, hidden_size*8)

        with vs.variable_scope("M1"):
            # Bidirectional GRU M1
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            blended_reps_1 = modeling_layer.build_graph(blended_reps, self.context_mask) # (batch_size, N, 2h)

        with vs.variable_scope("M2"):
            # Bidrectional GRU M2
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            blended_reps_2 = modeling_layer.build_graph(blended_reps_1, self.context_mask) # (batch_size, N, 2h)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            p1 = tf.layers.dense(tf.concat([blended_reps, blended_reps_1], axis=2), 1, use_bias=False) # (batch_size, N, 1)
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(p1, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            p2 = tf.layers.dense(tf.concat([blended_reps, blended_reps_2], axis=2), 1, use_bias=False) # (batch_size, N, 1)
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(p2, self.context_mask)

class BiDAFModel(QAModel):
    """BiDAF Question Answering module"""

    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        with vs.variable_scope("Encoder"):
        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BiAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, U_tilde, _, H_tilde = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens, self.context_mask) # shapes are each (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, U_tilde, context_hiddens * U_tilde, context_hiddens * H_tilde], axis=2) # (batch_size, context_len, hidden_size*8)

        with vs.variable_scope("M1_init"):
            # Bidirectional GRU M1
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            blended_reps_1_init = modeling_layer.build_graph(blended_reps, self.context_mask) # (batch_size, N, 2h)

        with vs.variable_scope("M1"):
            # Bidrectional GRU M2
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            blended_reps_1 = modeling_layer.build_graph(blended_reps_1_init, self.context_mask) # (batch_size, N, 2h)

        with vs.variable_scope("M2"):
            # Bidrectional GRU M2
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            blended_reps_2 = modeling_layer.build_graph(blended_reps_1, self.context_mask) # (batch_size, N, 2h)

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(tf.concat([blended_reps, blended_reps_1], axis=2), self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(tf.concat([blended_reps, blended_reps_2], axis=2), self.context_mask)
