"""
@author: Thang Nguyen <nhthang1009@gmail.com>
"""
import os
import shutil

import numpy as np
import tensorflow as tf

from src.utils import get_num_classes, create_dataset
from src.very_deep_cnn import Very_deep_cnn

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.flags.DEFINE_string("alphabet", """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
                       "Valid characters used for model")
tf.flags.DEFINE_string("train_set", "data/train.csv", "Path to the training set")
tf.flags.DEFINE_string("test_set", "data/test.csv", "Path to the test set")
tf.flags.DEFINE_integer("test_interval", 1, "Number of epochs between testing phases")
tf.flags.DEFINE_integer("max_length", 1024, "Maximum length of input")
tf.flags.DEFINE_integer("depth", 29, "9, 17, 29 or 49")
tf.flags.DEFINE_integer("batch_size", 128, "Minibatch size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs")
tf.flags.DEFINE_float("lr", 5e-4, "Learning rate")
tf.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.flags.DEFINE_string("log_path", "tensorboard/vdcnn", "path to tensorboard folder")
tf.flags.DEFINE_string("saved_path", "trained_models", "path to store trained model")

tf.flags.DEFINE_float("es_min_delta", 0,
                      "Early stopping's parameter: minimum change loss to qualify as an improvement")
tf.flags.DEFINE_integer("es_patience", 3,
                        "Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train():
    num_classes = get_num_classes(FLAGS.train_set)
    model = Very_deep_cnn(batch_size=FLAGS.batch_size, num_classes=num_classes, depth=FLAGS.depth,
                          num_embedding=len(FLAGS.alphabet))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True

        training_set, num_training_iters = create_dataset(FLAGS.train_set, FLAGS.alphabet, FLAGS.max_length,
                                                          FLAGS.batch_size, True)
        test_set, num_test_iters = create_dataset(FLAGS.test_set, FLAGS.alphabet, FLAGS.max_length, FLAGS.batch_size,
                                                  False)
        train_iterator = training_set.make_initializable_iterator()
        test_iterator = test_set.make_initializable_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        is_training = tf.placeholder(tf.bool, name='is_training')

        iterator = tf.data.Iterator.from_string_handle(handle, training_set.output_types, training_set.output_shapes)
        texts, labels = iterator.get_next()

        logits = model.forward(texts, is_training)
        loss = model.loss(logits, labels)
        loss_summary = tf.summary.scalar("loss", loss)
        accuracy = model.accuracy(logits, labels)
        accuracy_sumary = tf.summary.scalar("accuracy", accuracy)
        batch_size = tf.unstack(tf.shape(texts))[0]
        confusion = model.confusion_matrix(logits, labels)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            lr = tf.train.exponential_decay(FLAGS.lr, global_step,
                                            FLAGS.num_epochs * num_training_iters, 0.96,
                                            staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        merged = tf.summary.merge([loss_summary, accuracy_sumary])
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if os.path.isdir(FLAGS.log_path):
            shutil.rmtree(FLAGS.log_path)
        os.makedirs(FLAGS.log_path)
        if os.path.isdir(FLAGS.saved_path):
            shutil.rmtree(FLAGS.saved_path)
        os.makedirs(FLAGS.saved_path)
        output_file = open(FLAGS.saved_path + os.sep + "logs.txt", "w")
        output_file.write("Model's parameters: {}".format(FLAGS.flag_values_dict()))
        best_loss = 1e5
        best_epoch = 0
        with tf.Session(config=session_conf) as sess:
            train_writer = tf.summary.FileWriter(FLAGS.log_path + os.sep + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_path + os.sep + 'test')
            sess.run(init)
            for epoch in range(FLAGS.num_epochs):
                sess.run(train_iterator.initializer)
                sess.run(test_iterator.initializer)
                train_handle = sess.run(train_iterator.string_handle())
                test_handle = sess.run(test_iterator.string_handle())
                train_iter = 0
                while True:
                    try:
                        _, tr_loss, tr_accuracy, summary, step = sess.run(
                            [train_op, loss, accuracy, merged, global_step],
                            feed_dict={handle: train_handle, is_training: True})
                        print("Epoch: {}/{}, Iteration: {}/{}, Loss: {}, Accuracy: {}".format(
                            epoch + 1,
                            FLAGS.num_epochs,
                            train_iter + 1,
                            num_training_iters,
                            tr_loss, tr_accuracy))
                        train_writer.add_summary(summary, step)
                        train_iter += 1
                    except (tf.errors.OutOfRangeError, StopIteration):
                        break
                if epoch % FLAGS.test_interval == 0:
                    loss_ls = []
                    loss_summary = tf.Summary()
                    accuracy_ls = []
                    accuracy_summary = tf.Summary()
                    confusion_matrix = np.zeros([num_classes, num_classes], np.int32)
                    num_samples = 0
                    while True:
                        try:
                            test_loss, test_accuracy, test_confusion, samples = sess.run(
                                [loss, accuracy, confusion, batch_size],
                                feed_dict={handle: test_handle, is_training: False})
                            loss_ls.append(test_loss * samples)
                            accuracy_ls.append(test_accuracy * samples)
                            confusion_matrix += test_confusion
                            num_samples += samples
                        except (tf.errors.OutOfRangeError, StopIteration):
                            break

                    mean_test_loss = sum(loss_ls) / num_samples
                    loss_summary.value.add(tag='loss', simple_value=mean_test_loss)
                    test_writer.add_summary(loss_summary, epoch)
                    mean_test_accuracy = sum(accuracy_ls) / num_samples
                    accuracy_summary.value.add(tag='accuracy', simple_value=mean_test_accuracy)
                    test_writer.add_summary(accuracy_summary, epoch)

                    output_file.write(
                        "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                            epoch + 1, FLAGS.num_epochs,
                            mean_test_loss,
                            mean_test_accuracy,
                            confusion_matrix))
                    print("Epoch: {}/{}, Final loss: {}, Final accuracy: {}".format(epoch + 1, FLAGS.num_epochs,
                                                                                    mean_test_loss,
                                                                                    mean_test_accuracy))
                    if mean_test_loss + FLAGS.es_min_delta < best_loss:
                        best_loss = mean_test_loss
                        best_epoch = epoch
                        saver.save(sess, "{}/char_level_cnn".format(FLAGS.saved_path))
                    if epoch - best_epoch > FLAGS.es_patience > 0:
                        print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch,
                                                                                                             best_loss,
                                                                                                             best_epoch))
                        break

        output_file.close()


if __name__ == "__main__":
    train()
