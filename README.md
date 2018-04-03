# cs224n-squad
Adapted from the code for the Default Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018

Note: this code is adapted in part from the [Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford Machine Learning Group.

## About

This repo is a deep learning system for Machine Comprehension, as defined by the [Stanford Question Answering Dataset (SQuAD) challenge](https://rajpurkar.github.io/SQuAD-explorer/). The task is as follows: given a paragraph, and a question about that paragraph as inputs, answer the question correctly by providing the start and ending word within the context.

## Models

### Baseline

Our baseline model has three components: a RNN encoder layer, that encodes both the context and the question into hidden states, an attention layer, that combines the context and question representations, and an output layer, which applies a fully connected layer and then two separate softmax layers (one to get the start location, and one to get the end location of the answer span). All these modules can be found in ```modules.py```, and the code to connect them is in ```qa_model.py```.

### Bi-directional Attention Flow (BiDAF)

Our implementation follows the original BiDAF paper, titled ["Bidirectional Attention Flow for Machine Comprehension
"](https://arxiv.org/abs/1611.01603). We implemented the initial embedding layer, contextual embedding layer, bi-attention layer, modeling layer, and finally the output layer. More details can be found in our paper.

### Co-Attention

We adapt a co-attention layer from the recently published paper [“Dynamic coattention networks forquestion answering”](https://arxiv.org/abs/1611.01604). So that we can compare its performance to that of the bi-attention layer, we modify the layer to mimic the bi-attention layer’s structure, but still capture essence of co-attention rather than bi-attention.

### Self-Attention

We adapt a self-attention layer from the recently published paper [“r-net:  Machine Reading Com-prehension with Self-Matching Networks”](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). Similarly, we modify it to mimic our bi-attention layer.

## Evaluation

### Dev Data

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| single   | 66.2   | 76.1   |
| ensemble | 68.4   | 77.8   |

### Test Data

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| ensemble | 67.7   | 77.1   |

Refer to our paper for more details.
See [SQuAD Leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) to compare with other models.

## Installation

```
./get_started.sh
```

## Run

Activate the virtual environment: ```source activate squad```

Training a Model:

```
cd code # Change to code directory
python main.py --experiment_name=<EXPERIMENT_NAME> --mode=train --model_type=<MODEL_TYPE> # model defaults to bidaf use flags 'baseline', 'coattn', 'selfattn', or 'bidaf' as described above
```

Inspecting Output:

```
python main.py --experiment_name=<EXPERIMENT_NAME> --mode=show_examples
```

Running Official Eval on a tiny dev dataset from CodaLab:

```
cd cs224n-squad # Go to the root of the repository
cl download -o data/tiny-dev.json 0x4870af # Download the sanity check dataset

python code/main.py --experiment_name=<EXPERIMENT_NAME> --model_type=<MODEL_TYPE> --mode=official_eval \
--json_in_path=data/tiny-dev.json \
--ckpt_load_dir=experiments/<EXPERIMENT_NAME>/best_checkpoint
```

Then run the following to evaluate the predictions:

```
python code/evaluate.py data/tiny-dev.json predictions.json
```

To run the ensembling for multiple experiments, modify the `run_ensemble.sh` bash file and run:

```
./run_ensemble.sh tiny-dev.json predictions.json
```

## Paper

Our paper can be found [here](http://web.stanford.edu/class/cs224n/reports/6857497.pdf), and our poster can be found [here](https://ezshen.github.io/files/cs224n_final_poster.pdf).

