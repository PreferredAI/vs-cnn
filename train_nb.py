import os
import h5py
import numpy as np
from sklearn.naive_bayes import GaussianNB
from absl import app
from absl import flags
from data_generator import CITIES
import metrics

# Parameters
# ==================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "data/features",
                    """Path to data folder""")
flags.DEFINE_string("dataset", "user",
                    """Name of dataset (business or user)""")


def load_data(fpath):
  with h5py.File(fpath, 'r') as f:
    filepaths = f['filepaths'][:]
    features = f['features'][:]
    labels = f['labels'][:]

  factor_idx = -2 if FLAGS.dataset == 'user' else -3
  factors = []
  for fpath in filepaths:
    factor = fpath.decode().split('_')[factor_idx]
    factors.append(factor)

  return factors, features, labels


def to_onehot(x, dim=2):
  onehot = np.zeros((len(x), dim))
  onehot[np.arange(len(x)), x] = 1
  return onehot


def main(_):
  _, features, labels = load_data(os.path.join(FLAGS.data_dir, FLAGS.dataset, 'train.h5'))

  # Training
  model = GaussianNB()
  model.fit(features, labels)

  # Evaluation
  pointwise_results = []
  pairwise_results = []
  mae_results = []
  test_sizes = []

  for city in CITIES:
    factors, features, labels = load_data(
        os.path.join(FLAGS.data_dir, FLAGS.dataset, 'val_{}.h5'.format(city)))

    pd_probs = model.predict_proba(features)
    onehot_labels = to_onehot(labels)

    pointwise_results.append(metrics.pointwise(pds=pd_probs, gts=onehot_labels))
    pairwise_results.append(metrics.pairwise(pds=pd_probs, gts=onehot_labels, factors=factors))
    mae_results.append(metrics.mae(pds=pd_probs, gts=onehot_labels))
    test_sizes.append(len(labels))

  result_file = open('result_{}_nb.txt'.format(FLAGS.dataset), 'w')

  layout = '{:15} {:>10} {:>10} {:>10} {:>10}'
  print(layout.format('City', 'Size', 'Pointwise', 'Pairwise', 'MAE'))
  result_file.write(layout.format('City', 'Size', 'Pointwise', 'Pairwise', 'MAE\n'))
  print('-' * 59)
  result_file.write('-' * 59 + '\n')

  for city, pointwise, pairwise, mae, tsize in zip(CITIES, pointwise_results, pairwise_results, mae_results, test_sizes):
    print(layout.format(
      city, tsize, '{:.3f}'.format(pointwise), '{:.3f}'.format(pairwise), '{:.3f}'.format(mae)))
    result_file.write(layout.format(
      city, tsize, '{:.3f}'.format(pointwise), '{:.3f}'.format(pairwise), '{:.3f}\n'.format(mae)))

  test_sizes = np.asarray(test_sizes, dtype=np.int)
  total = np.sum(test_sizes)
  avg_pointwise = np.sum(np.asarray(pointwise_results) * test_sizes) / total
  avg_pairwise = np.sum(np.asarray(pairwise_results) * test_sizes) / total
  avg_mae = np.sum(np.asarray(mae_results) * test_sizes) / total
  print('-' * 59)
  result_file.write('-' * 59 + '\n')
  print(layout.format('Avg.', total, '{:.3f}'.format(avg_pointwise), '{:.3f}'.format(avg_pairwise),
                      '{:.3f}'.format(avg_mae)))
  result_file.write(
    layout.format('Avg.', total, '{:.3f}'.format(avg_pointwise), '{:.3f}'.format(avg_pairwise),
                  '{:.3f}\n'.format(avg_mae)))

  result_file.close()


if __name__ == '__main__':
  app.run(main)