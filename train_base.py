import os
import gpu_utils
gpu_utils.setup_one_gpu()

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model_base import VS_CNN
from data_generator import DataGenerator, CITIES
from datetime import datetime
from tqdm import trange
from tb_logger import Logger
import numpy as np
import metrics

from absl import app
from absl import flags

# Parameters
# ==================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "data",
                    """Path to data folder""")
flags.DEFINE_string("dataset", "user",
                    """Name of dataset (business or user)""")

flags.DEFINE_integer("num_checkpoints", 5,
                     """Number of checkpoints to store (default: 1)""")
flags.DEFINE_integer("num_epochs", 50,
                     """Number of training epochs (default: 50)""")
flags.DEFINE_integer("batch_size", 50,
                     """Batch Size (default: 50)""")
flags.DEFINE_integer("num_threads", 8,
                     """Number of threads for data processing (default: 2)""")
flags.DEFINE_integer("display_step", 10,
                     """Display after number of steps (default: 10)""")

flags.DEFINE_float("learning_rate", 0.0001,
                   """Learning rate (default: 0.0001)""")
flags.DEFINE_float("lambda_reg", 0.0005,
                   """Regularization lambda factor (default: 0.0005)""")
flags.DEFINE_float("dropout_keep_prob", 0.5,
                   """Probability of keeping neurons (default: 0.5)""")

flags.DEFINE_boolean("allow_soft_placement", True,
                     """Allow device soft device placement""")


def learning_rate_with_decay(initial_learning_rate, batches_per_epoch, boundary_epochs, decay_rates):
  # Reduce the learning rate at certain epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def loss_fn(model):
  with tf.name_scope("loss"):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=model.y, logits=model.fc8)
    l2_regularization = 0.5 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])
    loss = tf.reduce_mean(cross_entropy + FLAGS.lambda_reg * l2_regularization)
  return loss


def train_fn(loss, generator, finetune_layers, train_layers):
  var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in finetune_layers]
  var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

  grads1 = tf.gradients(loss, var_list1)
  grads2 = tf.gradients(loss, var_list2)

  global_step = tf.Variable(0, trainable=False)
  learning_rate = learning_rate_with_decay(initial_learning_rate=FLAGS.learning_rate,
                                           batches_per_epoch=generator.train_batches_per_epoch,
                                           boundary_epochs=[20, 40], decay_rates=[1, 0.1, 0.01])(global_step)
  opt1 = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
  opt2 = tf.train.MomentumOptimizer(10 * learning_rate, momentum=0.9)

  train_op1 = opt1.apply_gradients(zip(grads1, var_list1), name='finetune_op')
  train_op2 = opt2.apply_gradients(zip(grads2, var_list2), global_step, name='train_op')
  train_op = tf.group(train_op1, train_op2)

  return train_op2, train_op, learning_rate


def train(sess, model, generator, train_op, learning_rate, loss, epoch, logger):
  generator.load_train_set(sess)

  loop = trange(generator.train_batches_per_epoch, desc='Training')
  for step in loop:
    _, batch_img, batch_label, _ = generator.get_next(sess)
    _, _loss = sess.run([train_op, loss],
                        feed_dict={model.x: batch_img,
                                   model.y: batch_label,
                                   model.keep_prob: FLAGS.dropout_keep_prob})
    loop.set_postfix(loss=_loss)
    if step > 0 and step % FLAGS.display_step == 0:
      _step = epoch * generator.train_batches_per_epoch + step
      logger.log_scalar('cross_entropy', _loss, _step)
      logger.log_scalar('learning_rate', sess.run(learning_rate), _step)


def test(sess, model, generator, result_file):
  pointwise_results = []
  pairwise_results = []
  mae_results = []

  for city in CITIES:
    # Validate the model on the entire validation set
    generator.load_test_set(sess, city)
    pds = []
    gts = []
    factors = []
    for _ in trange(generator.test_batches_per_epoch[city], desc=city):
      _, batch_img, batch_label, batch_factor = generator.get_next(sess)
      pd = sess.run(model.prob, feed_dict={model.x: batch_img})
      pds.extend(pd.tolist())
      gts.extend(batch_label.tolist())
      factors.extend(batch_factor.tolist())

    pds = np.asarray(pds)
    gts = np.asarray(gts)

    pointwise_results.append(metrics.pointwise(pds, gts))
    pairwise_results.append(metrics.pairwise(pds, gts, factors))
    mae_results.append(metrics.mae(pds, gts))

  layout = '{:15} {:>10} {:>10} {:>10} {:>10}'
  print(layout.format('City', 'Size', 'Pointwise', 'Pairwise', 'MAE'))
  result_file.write(layout.format('City', 'Size', 'Pointwise', 'Pairwise', 'MAE\n'))
  print('-' * 59)
  result_file.write('-' * 59 + '\n')
  test_sizes = []
  for city, pointwise, pairwise, mae in zip(CITIES, pointwise_results, pairwise_results, mae_results):
    print(layout.format(
      city, generator.test_sizes[city], '{:.3f}'.format(pointwise), '{:.3f}'.format(pairwise), '{:.3f}'.format(mae)))
    result_file.write(layout.format(
      city, generator.test_sizes[city], '{:.3f}'.format(pointwise), '{:.3f}'.format(pairwise), '{:.3f}\n'.format(mae)))
    test_sizes.append(generator.test_sizes[city])

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
  result_file.flush()


def save_model(sess, saver, epoch, checkpoint_dir):
  print("{} Saving checkpoint of model...".format(datetime.now()))
  checkpoint_name = os.path.join(checkpoint_dir,
                                 'model_epoch' + str(epoch + 1) + '.ckpt')
  save_path = saver.save(sess, checkpoint_name)
  print("{} Model checkpoint saved at {}".format(datetime.now(), save_path))

  # Save trained weights for initializing factor models
  weight_dic = {}
  for layer in ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']:
    weight_dic[layer] = []
    with tf.variable_scope(layer, reuse=True):
      weight_dic[layer].extend(sess.run([tf.get_variable('weights'), tf.get_variable('biases')]))
  np.save('weights/{}_base.npy'.format(FLAGS.dataset), weight_dic)


def main(_):
  skip_layers = ['fc8']  # no pre-trained weights
  train_layers = ['fc8']
  finetune_layers = ['fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']

  writer_dir = "logs/{}".format(FLAGS.dataset)
  checkpoint_dir = "checkpoints/{}".format(FLAGS.dataset)

  if tf.gfile.Exists(writer_dir):
    tf.gfile.DeleteRecursively(writer_dir)
  tf.gfile.MakeDirs(writer_dir)

  if tf.gfile.Exists(checkpoint_dir):
    tf.gfile.DeleteRecursively(checkpoint_dir)
  tf.gfile.MakeDirs(checkpoint_dir)

  generator = DataGenerator(data_dir=FLAGS.data_dir,
                            dataset=FLAGS.dataset,
                            batch_size=FLAGS.batch_size,
                            num_threads=FLAGS.num_threads)

  model = VS_CNN(num_classes=2, skip_layers=skip_layers)
  loss = loss_fn(model)
  warm_up, train_op, learning_rate = train_fn(loss, generator, finetune_layers, train_layers)
  saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    logger = Logger(writer_dir, sess.graph)
    result_file = open('result_{}_base.txt'.format(FLAGS.dataset), 'w')

    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), writer_dir))
    for epoch in range(FLAGS.num_epochs):
      print("\n{} Epoch: {}/{}".format(datetime.now(), epoch + 1, FLAGS.num_epochs))
      result_file.write("\n{} Epoch: {}/{}\n".format(datetime.now(), epoch + 1, FLAGS.num_epochs))

      if epoch < 20:
        update_op = warm_up
      else:
        update_op = train_op
      train(sess, model, generator, update_op, learning_rate, loss, epoch, logger)

      test(sess, model, generator, result_file)
      save_model(sess, saver, epoch, checkpoint_dir)

    result_file.close()


if __name__ == '__main__':
  app.run(main)
