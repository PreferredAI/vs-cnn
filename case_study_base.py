import os
from shutil import copy2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gpu_utils
gpu_utils.setup_one_gpu()

import tensorflow as tf

from model_base import VS_CNN
from data_generator import DataGenerator, CITIES
from tqdm import trange
import numpy as np

from absl import app
from absl import flags

# Parameters
# ==================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "data",
                    """Path to data folder""")
flags.DEFINE_string("dataset", "user",
                    """Name of dataset (business or user)""")

flags.DEFINE_integer("batch_size", 50,
                     """Batch Size (default: 50)""")
flags.DEFINE_integer("num_threads", 8,
                     """Number of threads for data processing (default: 2)""")
flags.DEFINE_integer("num_images", 300,
                     """Number of images to be retrieved (default: 300)""")

flags.DEFINE_boolean("allow_soft_placement", True,
                     """Allow device soft device placement""")


def retrieve(sess, model, generator):
  pos_probs = []
  neg_probs = []

  for city in CITIES:
    # Validate the model on the entire validation set
    generator.load_test_set(sess, city)

    for _ in trange(generator.test_batches_per_epoch[city], desc=city):
      batch_fn, batch_img, batch_label, batch_factor = generator.get_next(sess)
      pd = sess.run(model.prob, feed_dict={model.x: batch_img})

      for fn, lb, prob in zip(batch_fn, batch_label, pd):
        gt_lb = int(lb.argmax())
        pd_lb = int(prob.argmax())
        if pd_lb == gt_lb == 0:
          neg_probs.append((fn, prob[0]))
        elif pd_lb == gt_lb == 1:
          pos_probs.append((fn, prob[1]))

  dtype = [('name', 'S200'), ('prob', float)]
  positive_image_prob = np.sort(np.array(pos_probs, dtype=dtype), order='prob')[::-1][:FLAGS.num_images]
  negative_image_prob = np.sort(np.array(neg_probs, dtype=dtype), order='prob')[::-1][:FLAGS.num_images]

  pos_save_dir = 'case_study/{}/most_positive'.format(FLAGS.dataset)
  neg_save_dir = 'case_study/{}/most_negative'.format(FLAGS.dataset)
  if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)
  if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)

  np.savetxt('case_study/{}/most_positive.txt'.format(FLAGS.dataset), positive_image_prob, fmt='%s %f')
  np.savetxt('case_study/{}/most_negative.txt'.format(FLAGS.dataset), negative_image_prob, fmt='%s %f')

  for i in range(FLAGS.num_images):
    # copy positive image to most_positive_image_prob folder
    fn = positive_image_prob[i]['name'].decode('utf-8')
    copy2(fn, os.path.join(pos_save_dir, fn.split('/')[-1]))

    # copy negative image to most_negative_image_prob folder
    fn = negative_image_prob[i]['name'].decode('utf-8')
    copy2(fn, os.path.join(neg_save_dir, fn.split('/')[-1]))


def main(_):
  generator = DataGenerator(data_dir=FLAGS.data_dir,
                            dataset=FLAGS.dataset,
                            batch_size=FLAGS.batch_size,
                            num_threads=FLAGS.num_threads)

  model = VS_CNN(num_classes=2, skip_layers=[], weights_path='weights/{}_base.npy'.format(FLAGS.dataset))

  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)
    retrieve(sess, model, generator)


if __name__ == '__main__':
  app.run(main)
