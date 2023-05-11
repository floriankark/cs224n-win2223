# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

"""
code copied from run.py and simplified
main change: remove all code related to predicting the location
and just predict "London" for every example
"""

from tqdm import tqdm
import argparse

import utils

argp = argparse.ArgumentParser()
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
args = argp.parse_args()

assert args.eval_corpus_path is not None

predictions = []
for line in tqdm(open(args.eval_corpus_path)):
    pred = "London"
    predictions.append(pred)
total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
else:
    print('Predictions written to {}; no targets provided'
            .format(args.outputs_path))

