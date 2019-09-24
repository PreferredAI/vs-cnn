import tensorflow as tf
import numpy as np
from layers import conv, lrn, max_pool, fc, dropout


class VS_CNN(object):

  def __init__(self, num_classes, skip_layers=None, weights_path='weights/bvlc_alexnet.npy'):
    self.num_classes = num_classes
    self.skip_layers = skip_layers
    self.weights_path = weights_path

    self._build_graph()

  def _build_graph(self):
    self.x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    self.y = tf.placeholder(tf.float32, [None, 2])
    self.keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')

    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) split into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool split into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
    dropout6 = dropout(fc6, self.keep_prob)

    # 7th Layer: FC (w ReLu) -> Dropout
    self.fc7 = fc(dropout6, 4096, 4096, name='fc7')
    dropout7 = dropout(self.fc7, self.keep_prob)

    # 8th Layer: FC and return unscaled activations
    self.fc8 = fc(dropout7, 4096, self.num_classes, relu=False, name='fc8')
    self.prob = tf.nn.softmax(self.fc8, name='prob')

  def load_initial_weights(self, session):
    weights_dict = dict(np.load(self.weights_path, allow_pickle=True, encoding='bytes').item())

    for op_name in weights_dict.keys():
      if op_name not in self.skip_layers:
        with tf.variable_scope(op_name, reuse=True):
          for data in weights_dict[op_name]:
            if len(data.shape) == 1:
              var = tf.get_variable('biases', trainable=True)
              session.run(var.assign(data))
            else:
              var = tf.get_variable('weights', trainable=True)
              session.run(var.assign(data))