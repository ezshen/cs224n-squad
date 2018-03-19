from collections import Counter
from itertools import groupby
import argparse
import json
from tqdm import tqdm
import random
from official_eval_helper import get_json_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-o', '--output_file', default='../ensemble_predictions.json')
    parser.add_argument("--json_in_path", default="../dev-v1.1.json")
    args = parser.parse_args()
    return args

def get_tokens(token_data)
    tokens = []
    while True:
        tok = readnext(token_data)
        if !tok:
            break
        tokens.append(tok)

def ensemble(args):
    qn_uuid_data, context_token_data, _ = get_json_data(args.json_in_path)

    context_tokens = get_tokens(context_token_data)
    qn_uuid_tokens = get_tokens(qn_uuid_data)
    uuid2context = dict(zip(qn_uuid_tokens, context_tokens))

    model_preds = {}
    for path in tqdm(args.paths):
        with open(path, 'r') as fh:
            preds = json.load(fh)
            for uuid, pred in preds.iteritems():
                model_preds.setdefault(uuid, []).append(pred)

    assert len(context_tokens) == len(qn_uuid_tokens)
    assert len(qn_uuid_tokens) == len(model_preds)

    final_pred = {}
    for uuid, preds in tqdm(model_preds.iteritems()):
        pred_start, pred_end = max_vote(preds)
        final_pred[uuid] = uuid2context[uuid][pred_start: pred_end+1]

def max_vote(preds):
    freq = groupby(Counter(preds).most_common(), lambda x:x[1])
    return random.choice([val for val,count in freq.next()[1]])

def main():
    args = get_args()
    ensemble(args)

if __name__ == "__main__":
    main()


