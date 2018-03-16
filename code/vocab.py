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

"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1


def get_glove(glove_char_path, glove_char_dim, glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_char_path:
      glove_char_dim:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      char_emb_matrix:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded
    char_vocab_size = int(94) # vocab size of char corpus
    char_emb_matrix = np.zeros((char_vocab_size + len(_START_VOCAB), glove_char_dim))
    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}
    char2id = {}
    id2char = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)
        char_emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_char_dim)

    # put start tokens in the dictionaries
    idx = 0
    char_idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        char2id[word] = char_idx
        id2char[char_idx] = word
        idx += 1
        char_idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    # go through glove_char vecs
    with open(glove_char_path, 'r') as fh:
        for line in tqdm(fh, total=char_vocab_size):
            line = line.lstrip().rstrip().split(" ")
            char = line[0]
            vector = list(map(np.float32, line[1:]))
            if glove_char_dim != len(vector):
                raise Exception("You set --glove_char_path=%s but --embedding_size=%i. If you set --glove_char_path yourself then make sure that --embedding_size matches!" % (glove_char_path, glove_char_dim))
            char_emb_matrix[char_idx, :] = vector
            char2id[char] = char_idx
            id2char[char_idx] = char
            char_idx += 1

    char_emb_matrix = char_emb_matrix.astype(np.float32)

    final_vocab_size = vocab_size + len(_START_VOCAB)
    final_char_vocab_size = char_vocab_size + len(_START_VOCAB)
    assert len(char2id) == final_char_vocab_size
    assert len(id2char) == final_char_vocab_size
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size
    assert char_idx == final_char_vocab_size

    return char_emb_matrix, char2id, id2char, emb_matrix, word2id, id2word
