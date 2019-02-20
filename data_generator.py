import os
import tensorflow as tf
import numpy as np

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

CITIES = ['Boston', 'Chicago', 'Houston', 'LosAngeles', 'NewYork', 'SanFrancisco', 'Seattle']


def parse_function(filename, label, factor, train=False):
  """Input parser for samples of the training set."""
  # convert label number into one-hot-encoding
  one_hot = tf.one_hot(label, 2)

  # load and preprocess the image
  img_string = tf.read_file(filename)
  img_decoded = tf.image.decode_jpeg(img_string, channels=3)

  img = tf.image.resize_images(img_decoded, [227, 227])
  img = tf.subtract(img, IMAGENET_MEAN)

  if train:
    """
    Data augmentation comes here.
    """
    img = tf.image.random_flip_left_right(img)

  # RGB -> BGR
  img_bgr = img[:, :, ::-1]

  return img_bgr, one_hot, factor

def parse_function_train(filename, label, factor):
  return parse_function(filename, label, factor, train=True)

def parse_function_test(filename, label, factor):
  return parse_function(filename, label, factor, train=False)

class DataGenerator(object):

  def __init__(self, data_dir, dataset, batch_size=1, num_threads=1, train_shuffle=False, buffer_size=10000):
    self.data_dir = data_dir
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.buffer_size = buffer_size
    self.factor_map = {}

    self.train_set, self.train_size = self._build_train_set(os.path.join(data_dir, dataset, 'train.txt'), train_shuffle)
    self.iterator = tf.data.Iterator.from_structure(self.train_set.output_types,
                                                    self.train_set.output_shapes)
    self.train_init_opt = self.iterator.make_initializer(self.train_set)
    self.next = self.iterator.get_next()
    self.train_batches_per_epoch = int(np.ceil(self.train_size / batch_size))

    self.test_sets = {}
    self.test_sizes = {}
    self.test_batches_per_epoch = {}
    self.test_inits = {}
    for city in CITIES:
      test_set, test_size = self._build_test_set(os.path.join(data_dir, dataset, 'val_{}.txt'.format(city)))
      self.test_sets[city] = test_set
      self.test_sizes[city] = test_size
      self.test_inits[city] = self.iterator.make_initializer(test_set)
      self.test_batches_per_epoch[city] = int(np.ceil(test_size / batch_size))

  def load_train_set(self, sess):
    sess.run(self.train_init_opt)

  def load_test_set(self, sess, city):
    sess.run(self.test_inits[city])

  def get_next(self, sess):
    return sess.run(self.next)

  @property
  def total_factors(self):
    return len(self.factor_map)

  @property
  def factor_id_map(self):
    return {k: v for v, k in self.factor_map.items()}

  def _parse_factor(self, img_path):
    tokens = img_path.split('/')[-1].split('_')
    if self.dataset == 'business':
      factor = '{}_{}'.format(tokens[0], tokens[2])
    elif self.dataset == 'user':
      factor = '{}_{}'.format(tokens[0], tokens[3])
    else:
      raise ValueError('Invalid dataset: %s.' % self.dataset)

    factor_id = self.factor_map.setdefault(factor, len(self.factor_map))
    return factor_id

  def _read_txt_file(self, data_file):
    """Read the content of the text file and store it into lists."""
    print('Loading data file: %s' % data_file)
    img_paths = []
    labels = []
    factors = []
    with open(data_file, 'r') as f:
      for line in f.readlines():
        items = line.split(' ')
        img_paths.append(os.path.join(self.data_dir, self.dataset, items[0]))
        labels.append(int(items[1]))
        factors.append(self._parse_factor(items[0]))
    return img_paths, labels, factors

  def _build_data_set(self, img_paths, labels, factors, map_fn, shuffle=False):
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    factors = tf.convert_to_tensor(factors, dtype=tf.int32)
    data = tf.data.Dataset.from_tensor_slices((img_paths, labels, factors))
    if shuffle:
      data = data.shuffle(buffer_size=self.buffer_size)
    data = data.map(map_fn, num_parallel_calls=self.num_threads)
    data = data.batch(self.batch_size)
    data = data.prefetch(self.num_threads)
    return data

  def _build_train_set(self, train_file, train_shuffle):
    train_img_paths, train_labels, train_factors = self._read_txt_file(train_file)
    return self._build_data_set(train_img_paths,
                                train_labels,
                                train_factors,
                                parse_function_train,
                                shuffle=train_shuffle), len(train_labels)

  def _build_test_set(self, test_file):
    test_img_paths, test_labels, test_factors = self._read_txt_file(test_file)
    return self._build_data_set(test_img_paths,
                                test_labels,
                                test_factors,
                                parse_function_test), len(test_labels)


