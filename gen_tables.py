import os
import argparse

import re

pattern = re.compile(r'\s+')

MODELS = ['nb', 'base', 'conv1', 'conv3', 'conv5', 'fc7']
METRICS = ['Pointwise', 'Pairwise', 'MAE']
CITIES = ['Boston', 'Chicago', 'Houston', 'LosAngeles', 'NewYork', 'SanFrancisco', 'Seattle', 'Avg.']


def read_file(fpath):
  result = {}
  with open(fpath, 'r') as f:
    for line in f:
      line = re.sub(pattern, '\t', line)
      tokens = line.strip().split('\t')
      if tokens[0] in CITIES:
        result[tokens[0]] = tokens[2:]
  return result


def read_results(result_dir, dataset):
  results = {}
  for model in MODELS:
    results[model] = read_file(os.path.join(result_dir, 'result_{}_{}.txt'.format(dataset, model)))
  return results


def gen_table(results, idx, fpath):
  layout = '{:16} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n'
  with open(fpath, 'w') as f:
    f.write(layout.format('City', 'NB', 'VS-CNN', 'Conv1', 'Conv3', 'Conv5', 'FC7'))
    f.write('-' * 70 + '\n')
    for city in CITIES:
      if city == CITIES[-1]:
        f.write('-' * 70 + '\n')
      f.write(layout.format(*([city] + [results[model][city][idx] for model in MODELS])))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--dir', type=str, default='.',
                      help='Path to directory containing result files')
  args = parser.parse_args()

  business_results = read_results(args.dir, 'business')
  user_results = read_results(args.dir, 'user')

  gen_table(business_results, 0, os.path.join(args.dir, 'table_1.txt'))
  gen_table(business_results, 1, os.path.join(args.dir, 'table_2.txt'))
  gen_table(business_results, 2, os.path.join(args.dir, 'table_3.txt'))

  gen_table(user_results, 0, os.path.join(args.dir, 'table_5.txt'))
  gen_table(user_results, 1, os.path.join(args.dir, 'table_6.txt'))
  gen_table(user_results, 2, os.path.join(args.dir, 'table_7.txt'))
