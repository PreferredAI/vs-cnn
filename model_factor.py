import os
import tensorflow as tf
import numpy as np
from layers import conv, lrn, max_pool, fc


class FVS_CNN(object):

  def __init__(self, num_classes, num_factors, factor_id_map, factor_layer='fc7',
               skip_layers=None, weights_path='weights/bvlc_alexnet.npy'):
    self.num_classes = num_classes
    self.num_factors = num_factors
    self.factor_layer = factor_layer
    self.factor_weight_size = self._get_weight_size(factor_layer, num_factors)
    self.skip_layers = skip_layers
    self.weights_path = weights_path

    self.factor_id_map = factor_id_map
    self.factor_weight_dict = {}
    self.factor_bias_dict = {}

    self._build_graph()

  def _get_weight_size(self, factor_layer, num_factors):
    if factor_layer == 'conv1':
      return [11, 11, 3, num_factors]
    elif factor_layer == 'conv3':
      return [3, 3, 128, num_factors]
    elif factor_layer == 'conv5':
      return [3, 3, 192, num_factors]
    elif factor_layer == 'fc7':
      return [4096, num_factors]
    else:
      raise ValueError('Invalid factor layer name: {}'.format(factor_layer))

  def _build_graph(self):
    self.x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    self.y = tf.placeholder(tf.float32, [None, 2])

    if self.factor_layer == 'conv1':
      self.factor_weights = tf.placeholder(tf.float32, shape=[11, 11, 3, self.num_factors])
    elif self.factor_layer == 'conv3':
      self.factor_weights = tf.placeholder(tf.float32, shape=[3, 3, 128, self.num_factors])
    elif self.factor_layer == 'conv5':
      self.factor_weights = tf.placeholder(tf.float32, shape=[3, 3, 192, self.num_factors])
    elif self.factor_layer == 'fc7':
      self.factor_weights = tf.placeholder(tf.float32, shape=[4096, self.num_factors])

    self.factor_biases = tf.placeholder(tf.float32, shape=[self.num_factors])

    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    if self.factor_layer == 'conv1':
      conv1_factor = conv(self.x, 11, 11, self.num_factors, 4, 4, padding='VALID', name='conv1_factor')
      conv1_shared = conv(self.x, 11, 11, 96 - self.num_factors, 4, 4, padding='VALID', name='conv1_shared')
      splits = tf.split(axis=3, num_or_size_splits=2, value=conv1_factor)
      conv1 = tf.concat([splits[0], conv1_shared, splits[1]], axis=3, name='conv1_concat')
    else:
      conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')

    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    if self.factor_layer == 'conv3':
      conv3_factor = conv(pool2, 3, 3, self.num_factors, 1, 1, groups=2, name='conv3_factor')
      conv3_shared = conv(pool2, 3, 3, 384 - self.num_factors, 1, 1, name='conv3_shared')
      splits = tf.split(axis=3, num_or_size_splits=2, value=conv3_factor)
      conv3 = tf.concat([splits[0], conv3_shared, splits[1]], axis=3, name='conv3_concat')
    else:
      conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) split into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool split into two groups
    if self.factor_layer == 'conv5':
      conv5_factor = conv(conv4, 3, 3, self.num_factors, 1, 1, groups=2, name='conv5_factor')
      conv5_shared = conv(conv4, 3, 3, 256 - self.num_factors, 1, 1, groups=2, name='conv5_shared')
      splits = tf.split(axis=3, num_or_size_splits=2, value=conv5_factor)
      conv5 = tf.concat([splits[0], conv5_shared, splits[1]], axis=3, name='conv5_concat')
    else:
      conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')

    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu)
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')

    if self.factor_layer == 'fc7':
      fc7_factor = fc(fc6, 4096, self.num_factors, 'fc7_factor')
      fc7_shared = fc(fc6, 4096, 4096 - self.num_factors, 'fc7_shared')
      fc7 = tf.concat([fc7_factor, fc7_shared], axis=1, name='fc7_concat')
    else:
      fc7 = fc(fc6, 4096, 4096, name='fc7')

    with tf.variable_scope('{}_factor'.format(self.factor_layer), reuse=True):
      self.assign_factor = tf.group(tf.get_variable('weights').assign(self.factor_weights),
                                    tf.get_variable('biases').assign(self.factor_biases))

    # 8th Layer: FC and return unscaled activations
    self.fc8 = fc(fc7, 4096, self.num_classes, relu=False, name='fc8')
    self.prob = tf.nn.softmax(self.fc8, name='prob')

  def load_factor_weights(self, session, factor_ids, weight_dir=None):
    factor_id = factor_ids[0]
    if factor_id in self.factor_weight_dict.keys():
      weights = self.factor_weight_dict[factor_id]
      biases = self.factor_bias_dict[factor_id]
    else:
      factor = self.factor_id_map[factor_id]
      if weight_dir:
        weights = np.load(os.path.join(weight_dir, factor, 'weights.npy'))
        biases = np.load(os.path.join(weight_dir, factor, 'biases.npy'))
      else:
        if self.factor_layer.startswith('conv'):
          weights = np.random.normal(0.0, scale=0.01, size=self.factor_weight_size)
        else:
          weights = np.random.normal(0.0, scale=0.005, size=self.factor_weight_size)
        biases = np.zeros(self.num_factors)
      self.factor_weight_dict[factor_id] = weights
      self.factor_bias_dict[factor_id] = biases
    session.run(self.assign_factor, feed_dict={self.factor_weights: weights,
                                               self.factor_biases: biases})

  def update_factor_weights(self, session, factor_ids):
    factor_id = factor_ids[0]
    with tf.variable_scope('{}_factor'.format(self.factor_layer), reuse=True):
      self.factor_weight_dict[factor_id], self.factor_bias_dict[factor_id] = \
        session.run([tf.get_variable('weights'), tf.get_variable('biases')])

  def load_initial_weights(self, session):
    print('Loading initial weights from:', self.weights_path)
    self.weights_dict = dict(np.load(self.weights_path, encoding='bytes').item())

    for op_name in self.weights_dict.keys():
      if op_name not in self.skip_layers:
        print('Loading weights for layer:', op_name)
        with tf.variable_scope(op_name, reuse=True):
          for data in self.weights_dict[op_name]:
            if len(data.shape) == 1:
              var = tf.get_variable('biases', trainable=True)
              session.run(var.assign(data))
            else:
              var = tf.get_variable('weights', trainable=True)
              session.run(var.assign(data))

    print('Loading weights for layer: {}_shared'.format(self.factor_layer))
    if self.factor_layer.startswith('conv'):
      with tf.variable_scope('{}_shared'.format(self.factor_layer), reuse=True):
        for data in self.weights_dict[self.factor_layer]:
          if len(data.shape) == 1:
            var = tf.get_variable('biases', trainable=True)
            session.run(var.assign(data[int(self.num_factors / 2):-int(self.num_factors / 2)]))
          else:
            var = tf.get_variable('weights', trainable=True)
            session.run(var.assign(data[:, :, :, int(self.num_factors / 2):-int(self.num_factors / 2)]))

    elif self.factor_layer == 'fc7':
      with tf.variable_scope('fc7_shared', reuse=True):
        for data in self.weights_dict['fc7']:
          if len(data.shape) == 1:
            var = tf.get_variable('biases', trainable=True)
            session.run(var.assign(data[self.num_factors:]))
          else:
            var = tf.get_variable('weights', trainable=True)
            session.run(var.assign(data[:, self.num_factors:]))