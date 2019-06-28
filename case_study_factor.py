import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import Counter
import glob
from shutil import copy2

import gpu_utils

gpu_utils.setup_one_gpu()

import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from model_factor import FVS_CNN
from data_generator import DataGenerator, parse_function_test
from tqdm import tqdm

# Parameters
# ==================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "data",
                    """Path to data folder""")
flags.DEFINE_string("dataset", "user",
                    """Name of dataset (business or user)""")
flags.DEFINE_string("input_dir", "",
                    """Path to input folder of selected images""")
flags.DEFINE_string("output_dir", "",
                    """Path to output folder of contrarian users/items""")

flags.DEFINE_string("factor_layer", "fc7",
                    """Name of layer to place the factors [conv1, conv3, conv5, fc7] (default: fc7)""")
flags.DEFINE_integer("num_factors", 16,
                     """Number of specific neurons for user/item (default: 16)""")
flags.DEFINE_integer("num_threads", 8,
                     """Number of threads for data processing (default: 2)""")
flags.DEFINE_integer("batch_size", 50,
                     """Batch Size (default: 50)""")
flags.DEFINE_integer("num_items", 100,
                     """Number of top user/business to be retrieved (default: 100)""")

flags.DEFINE_boolean("allow_soft_placement", True,
                     """Allow device soft device placement""")


def parse_fn(fn, factor_map):
  tokens = fn.split('/')[-1].split('_')
  if tokens[1] == 'p':
    label = 1
  else:
    label = 0

  if FLAGS.dataset == 'business':
    factor = '{}_{}'.format(tokens[0], tokens[2])
  else:
    factor = '{}_{}'.format(tokens[0], tokens[3])
  factor_id = factor_map[factor]

  return label, factor_id


def load_input_images(factor_map):
  labels = []
  factors = []

  files = [f for f in glob.glob(os.path.join(FLAGS.input_dir, '*.jpg'))]
  for f in files:
    label, factor_id = parse_fn(f, factor_map)
    labels.append(label)
    factors.append(factor_id)

  return files, labels, factors


def build_iter(img_paths, labels, factors):
  img_tensor = tf.convert_to_tensor(img_paths, dtype=tf.string)
  label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
  factor_tensor = tf.convert_to_tensor(factors, dtype=tf.int32)

  data = tf.data.Dataset.from_tensor_slices((img_tensor, label_tensor, factor_tensor))
  data = data.map(parse_function_test, num_parallel_calls=FLAGS.num_threads)
  data = data.batch(FLAGS.batch_size)
  data = data.prefetch(FLAGS.num_threads)

  iter = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
  init_op = iter.make_initializer(data)
  next_op = iter.get_next()

  return init_op, next_op


def load_factor_weights(model, weight_dir):
  for factor_id, factor in tqdm(model.factor_id_map.items(), desc='Loading weights'):
    weights = np.load(os.path.join(weight_dir, factor, 'weights.npy'), allow_pickle=True)
    biases = np.load(os.path.join(weight_dir, factor, 'biases.npy'), allow_pickle=True)
    model.factor_weight_dict[factor_id] = weights
    model.factor_bias_dict[factor_id] = biases


def retrieve(sess, model, generator):
  img_paths, labels, factors = load_input_images(generator.factor_map)
  num_batches = int(np.ceil(len(img_paths) / FLAGS.batch_size))

  init_op, next_op = build_iter(img_paths, labels, factors)
  reverse_count = Counter()

  for factor_id in tqdm(model.factor_id_map.keys(), desc="Retrieving"):
    model.load_factor_weights(sess, [factor_id], "weights/{}".format(FLAGS.dataset))
    sess.run(init_op)
    pds = []
    gts = []
    for _ in range(num_batches):
      _, batch_img, batch_label, batch_factor = sess.run(next_op)
      pd = sess.run(model.prob, feed_dict={model.x: batch_img})
      pds.extend(pd.argmax(axis=1).tolist())
      gts.extend(batch_label.argmax(axis=1).tolist())
    pds = np.asarray(pds)
    gts = np.asarray(gts)

    reverse_count[model.factor_id_map[factor_id]] = np.sum(np.abs(gts - pds))

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  retrieved_sentiment = 'p' if labels[0] == 0 else 'n'

  with open(os.path.join(FLAGS.output_dir, 'reverse_items.txt'), 'w') as f:
    for factor, count in reverse_count.most_common(FLAGS.num_items):
      f.write('{} {}\n'.format(factor, count))

      dst_dir = os.path.join(FLAGS.output_dir, factor)
      if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
      city, item_name = factor.split('_')
      src_dir = '{}/{}/train'.format(FLAGS.data_dir, FLAGS.dataset)
      for fn in [f for f in os.listdir(src_dir)
                 if '{}_{}'.format(city, retrieved_sentiment) in f
                    and item_name + '_' in f]:
        copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))


def main(_):
  generator = DataGenerator(data_dir=FLAGS.data_dir,
                            dataset=FLAGS.dataset,
                            batch_size=1,
                            num_threads=FLAGS.num_threads)

  model = FVS_CNN(num_classes=2, num_factors=FLAGS.num_factors, factor_id_map=generator.factor_id_map,
                  factor_layer=FLAGS.factor_layer)
  saver = tf.train.Saver()

  # Start Tensorflow session
  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint("checkpoints/f_{}".format(FLAGS.dataset)))
    load_factor_weights(model, "weights/{}".format(FLAGS.dataset))
    retrieve(sess, model, generator)


if __name__ == '__main__':
  app.run(main)