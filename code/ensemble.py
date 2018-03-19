from collections import Counter
from itertools import groupby
import argparse
import json
from tqdm import tqdm
import random
from official_eval_helper import get_json_data, readnext
import io

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--output_file', default='ensemble_predictions.json')
    parser.add_argument("--json_in_path", default="dev-v1.1.json")
    args = parser.parse_args()
    return args

def get_tokens(token_data):
    tokens = []
    while True:
        tok = readnext(token_data)
        if not tok:
            break
        tokens.append(tok)
    return tokens

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
                model_preds.setdefault(uuid, []).append(tuple(pred))

    assert len(context_tokens) == len(qn_uuid_tokens)
    assert len(qn_uuid_tokens) == len(model_preds)

    final_pred = {}
    for uuid, preds in tqdm(model_preds.iteritems()):
	pred_start, pred_end = max_vote(preds)
        final_pred[uuid] = " ".join(uuid2context[uuid][pred_start: pred_end+1])

    print "writing output file..."
    with io.open(args.output_file, 'w', encoding='utf-8') as f:
    	f.write(unicode(json.dumps(final_pred, ensure_ascii=False)))

def max_vote(preds):
    freq = groupby(Counter(preds).most_common(), lambda x:x[1])
    return random.choice([val for val,count in freq.next()[1]])

def main():
    args = get_args()
    ensemble(args)

if __name__ == "__main__":
    main()


