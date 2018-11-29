import os
import tensorflow as tf
from model_factor import FVS_CNN
from data_generator import DataGenerator
from tqdm import tqdm


# Parameters
# ==================================================
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir", "data",
                           """Path to data folder""")
tf.app.flags.DEFINE_string("dataset", "user",
                           """Name of dataset (business or user)""")
tf.app.flags.DEFINE_integer("num_factors", 16,
                            """Number of specific neurons for user/item (default: 16)""")
tf.app.flags.DEFINE_integer("num_threads", 2,
                            """Number of threads for data processing (default: 2)""")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            """Allow device soft device placement""")


cities = ['Boston', 'Chicago', 'Houston', 'LosAngeles', 'NewYork', 'SanFrancisco', 'Seattle']


# Network params
num_classes = 2

# Path to model checkpoints
checkpoint_dir = "checkpoints/f_{}".format(FLAGS.dataset)
weight_dir = "weights/{}".format(FLAGS.dataset)


def eval_fn(model):
  # Evaluation op: Accuracy of the model
  with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(model.fc8, 1), tf.argmax(model.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy


def sort_neg_img(pos_imgs, neg_imgs):
  sorted_neg_imgs = []
  for pos_img in pos_imgs:
    if FLAGS.dataset == 'business':
      factor = pos_img.split('_')[-3] + '_'
    else: # user dataset
      factor = pos_img.split('_')[-2] + '_'
    for neg_img in neg_imgs:
      if factor in neg_img:
        sorted_neg_imgs.append(neg_img)
        neg_imgs.remove(neg_img)
        break
  return sorted_neg_imgs


def pairwise_test(img_prob_dic, pos_imgs, neg_imgs):
  neg_imgs = sort_neg_img(pos_imgs, neg_imgs)

  acc = 0.
  for pos_img, neg_img in zip(pos_imgs, neg_imgs):
    pos_img_prob = img_prob_dic[pos_img]
    neg_img_prob = img_prob_dic[neg_img]

    if pos_img_prob > neg_img_prob:
      acc += 1.0
    elif pos_img_prob == neg_img_prob:
      acc += 0.5

  return acc / len(pos_imgs)



def test(sess, model, generator, accuracy, probs):
  generator.load_test_set(sess)

  pointwise_acc = 0.
  count = 0
  img_prob_dic = {}
  pos_imgs = []
  neg_imgs = []

  for b in tqdm(range(generator.test_batches_per_epoch), 'Testing'):
    img, label, factor = generator.get_next(sess)
    model.load_factor_weights(sess, factor, weight_dir)
    acc, _probs = sess.run([accuracy, probs],
                           feed_dict={model.x: img, model.y: label})
    pointwise_acc += acc
    count += 1

    # Pairwise accuracy
    img_path = generator.test_img_paths[b]
    img_prob_dic[img_path] = _probs[0][1]

    if int(label[0][1]) == 1:
      pos_imgs.append(img_path)
    else:
      neg_imgs.append(img_path)

  pointwise_acc /= count
  print("Pointwise Accuracy = {:.3f}".format(pointwise_acc))

  pairwise_acc = pairwise_test(img_prob_dic, pos_imgs, neg_imgs)
  print("Pairwise Accuracy = {:.3f}".format(pairwise_acc))

  return pointwise_acc, pairwise_acc, count



def main(_):
  # Initialize model
  model = FVS_CNN(num_classes, FLAGS.num_factors)
  accuracy = eval_fn(model)
  probs = tf.nn.softmax(logits=model.fc8)

  # Start Tensorflow session
  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    total_pointwise = 0.
    total_pairwise = 0.
    total_count = 0

    # Loop over number of epochs
    for city in cities:
      print('\n' + city)
      val_file = os.path.join(FLAGS.data_dir, FLAGS.dataset, 'val_{}.txt'.format(city))

      # Place data loading and preprocessing on the cpu
      with tf.device('/cpu:0'):
        generator = DataGenerator(data_dir=FLAGS.data_dir,
                                  dataset=FLAGS.dataset,
                                  batch_size=1,
                                  test_file=val_file,
                                  num_threads=FLAGS.num_threads)

      pointwise_acc, pairwise_acc, count = test(sess, model, generator, accuracy, probs)
      total_pointwise += pointwise_acc * count
      total_pairwise += pairwise_acc * count
      total_count += count

    print('\nAvg. Pointwise Accuracy = {:.3f}'.format(total_pointwise / total_count))
    print('Avg. Pointwise Accuracy = {:.3f}\n'.format(total_pairwise / total_count))

if __name__ == '__main__':
  tf.app.run()
