from collections import Counter
from itertools import groupby
import numpy as np
import argparse
import json
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    # parser.add_argument('-o', '--output_file', default='ensemble_predictions.json')
    # parser.add_argument("--data_path", default="data/squad/data_test.json")
    # parser.add_argument("--shared_path", default="data/squad/shared_test.json")
    args = parser.parse_args()
    return args

def ensemble(args):
	model_preds = []
	for path in tqdm(args.paths):
		with open(path, 'r') as pred:
			model_preds.append(json.load(pred))





# def final_start_end(start_pos, end_pos):
# 	final_start = np.zeros(start_pos.shape[0])
# 	final_end = np.zeros(end_pos.shape[0])
# 	for i in range(len(final_start)):
# 		possible_start_pos = start_pos[i]
# 		possible_end_pos = end_pos[i]
# 		freqs_start = groupby(Counter(possible_start_pos).most_common(), lambda x:x[1])
# 		start_choices = np.asarray([val for val,count in freqs_start.next()[1]])
# 		freqs_end = groupby(Counter(possible_end_pos).most_common(), lambda x:x[1])
# 		end_choices = np.asarray([val for val,count in freqs_end.next()[1]])
# 		if len(end_choices) >= len(start_choices):
# 			start = np.random.choice(start_choices)
# 			possible_end_pos = possible_end_pos[np.where(possible_start_pos == start)]
# 			freqs_end = groupby(Counter(possible_end_pos).most_common(), lambda x:x[1])
# 			end = np.random.choice(np.asarray([val for val,count in freqs_end.next()[1]]))
# 		else:
# 			end = np.random.choice(end_choices)
# 			possible_start_pos = possible_start_pos[np.where(possible_end_pos == end)]
# 			freqs_start = groupby(Counter(possible_start_pos).most_common(), lambda x:x[1])
# 			start = np.random.choice(np.asarray([val for val,count in freqs_start.next()[1]]))
# 		final_start[i], final_end[i] = start, end
# 	return final_start, final_end

# start_pos = np.asarray([[3, 4, 4, 4, 4, 7, 3, 3, 3,], [5, 5, 6, 7, 8, 1, 2, 3, 4], [1, 1, 2, 2, 3, 3, 4, 4, 5]])
# end_pos = np.asarray([[5, 4, 4, 4, 4, 5, 5, 5, 5], [2, 2, 3, 3, 1, 4, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]])

# print final_start_end(start_pos, end_pos)


def main():
    args = get_args()
    ensemble(args)

if __name__ == "__main__":
    main()


