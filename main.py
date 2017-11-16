import tensorflow as tf
import numpy as np
import os
import csv
import time
import glob
import model
from utils import loc2array, preprocessing, print_result

flags = tf.app.flags

# Directories
flags.DEFINE_string('data_dir', './data/', 'Directory to store input dataset')
flags.DEFINE_string('result_dir', './figure/', 'Directory to store result')

# Run Settings
flags.DEFINE_string('input_file', 'loc', 'Input file: loc')
flags.DEFINE_boolean('test', False, 'Run Test or Not')
flags.DEFINE_string('test_file', 'E100_BS50_S40_FC32_L32', 'Test checkpoint filename')

# Model Settings
flags.DEFINE_integer('input_dim', 2, 'The dimension of input')
flags.DEFINE_integer('num_class', 3, 'The number of class for Soft Sensor Localization')
flags.DEFINE_integer('seq_length', 40, 'The size of window for LSTM network')
flags.DEFINE_integer('num_lstm_layer', 3, 'The number of multi-LSTM layers')
flags.DEFINE_integer('lstm_unit', 32, 'The size of hidden unit in a LSTM layer')
flags.DEFINE_integer('fc_hidden_unit', 32, 'The size of hidden unit in a fully-connected layer')

# Training & Optimizer
flags.DEFINE_integer('total_epoch', 100, 'The number of training epoch')
flags.DEFINE_integer('batch_size', 50, 'The size of batch for minibatch training')
flags.DEFINE_float('learning_rate', 0.001, 'The learning rate of training')

# Debug & Etcs
flags.DEFINE_integer('random_seed', 2345, 'Value of random seed')

FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
current_time = time.strftime('%m%d', time.localtime(time.time()))


def run_train(sess, train_file):
    """Training the model
    """
    sensor_data = []
    sensor_loc = []

    with open(train_file, "r") as f:
        header = f.readline()
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            x_location = int(row[1]) // 15 + 2
            sensor_data.append([int(row[0]), float(row[5]), float(row[6]), float(row[7])])  # Trial, V, Grad, Ref_force
            sensor_loc.append(loc2array(FLAGS.num_class, x_location))

    train_seq = preprocessing(sensor_data)
    total_batch = np.shape(train_seq)[0] // FLAGS.batch_size

    loc_input = np.array(sensor_data)[:, (1, 2)]  # V, gradV
    sensor_loc = np.array(sensor_loc)  # Location of pressure
    sensor_output = np.array(sensor_data)[:, 3]  # Ref_Force

    X = tf.get_collection('input')[0]
    V = tf.get_collection('input')[1]
    L = tf.get_collection('ground_truth')[0]
    Y = tf.get_collection('ground_truth')[1]

    for epoch in range(1, FLAGS.total_epoch+1):
        avg_cost = 0.

        # Training step
        for i in range(total_batch):
            data_idxs = train_seq[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
            seq_idxs = np.array([data_idxs - n for n in reversed(range(0, FLAGS.seq_length))]).T

            seq_x = np.reshape(loc_input[seq_idxs], [-1, FLAGS.seq_length, FLAGS.input_dim])
            v = np.reshape(np.array(sensor_data)[:, 1][data_idxs], [-1, 1])
            seq_l = np.reshape(sensor_loc[data_idxs], [-1, FLAGS.num_class])
            seq_y = np.reshape(sensor_output[data_idxs], [-1, 1])

            _, _cost = sess.run(tf.get_collection('train_ops'), feed_dict={X: seq_x, V: v, L: seq_l, Y: seq_y})
            avg_cost += _cost / total_batch
        print("Epoch: {}, Cost: {:.4}".format(epoch, avg_cost))
    print("Localization - Optimization Finished!")


def run_test(sess, test_file):
    """Testing the model
    """
    test_data = []
    test_options = []

    X = tf.get_collection('input')[0]
    V = tf.get_collection('input')[1]

    with open(test_file, "r") as f:
        header = f.readline()
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            x_location = int(row[1]) // 15 + 2
            test_data.append([int(row[0]), float(row[5]), float(row[6]), float(row[7])])  # Trial, V, Grad, Ref_force
            test_options.append([int(row[0]), x_location, float(row[3]), float(row[4]), float(row[5]), float(row[7])]) # Trial, x_location, z_location, velocity, voltage, Ref_force


    # Data Transformations
    test_seq = np.array(range(FLAGS.seq_length, np.shape(test_data)[0]))
    test_idxs = np.array([test_seq - n for n in reversed(range(0, FLAGS.seq_length))]).T

    # Test Inputs
    test_input = np.array(test_data)[:, (1, 2)]  # V, gradV
    test_input = np.reshape(test_input[test_idxs], [-1, FLAGS.seq_length, FLAGS.input_dim])
    test_V = np.reshape(np.array(test_data)[:, 1][test_seq], [-1, 1])

    # Test Run
    estimate_loc, estimate_force = sess.run(tf.get_collection('test_ops'), feed_dict={X: test_input, V: test_V})

    estimate_force = np.reshape(estimate_force, [-1])
    estimate_loc_prob = np.reshape(estimate_loc, (-1, FLAGS.num_class)).tolist()

    test_options = np.array(test_options)
    test_options = np.reshape(test_options[test_seq], [-1, 6])

    test_force_truth = np.array(test_data)[test_seq, 3]
    test_loc_truth = test_options[:, 1]
    test_loc_truth = np.reshape(test_loc_truth, [-1, 1])

    print_result(test_options, estimate_force, test_force_truth, estimate_loc, test_loc_truth, test_seq)


def main(argv=None):
    """Main Function
    """

    if FLAGS.test:
        args = FLAGS.test_file.split("_")

        FLAGS.batch_size = int(args[1][2:])
        FLAGS.seq_length = int(args[2][1:])
        FLAGS.fc_hidden_unit = int(args[3][2:])
        FLAGS.lstm_unit = int(args[4][1:])


    model.build()

    train_file = FLAGS.data_dir + FLAGS.input_file + "_train.csv"
    test_file = FLAGS.data_dir + FLAGS.input_file + "_test.csv"

    with tf.Session() as sess:
        # ckpt (checkpoint) saver
        saver = tf.train.Saver()
        if FLAGS.test:
            test_ckpt_dir = './ckpt/' + FLAGS.input_file + '/' + FLAGS.test_file + '/'
            check = test_ckpt_dir + "*.meta"
            ckpt_idx = len(glob.glob(check))

            for i in range(ckpt_idx):
                test_ckpt_path = test_ckpt_dir + "train_result" + str(i+1) + ".ckpt"
                saver.restore(sess, test_ckpt_path)
                print ('Restored variables from %s.' % test_ckpt_path)
                run_test(sess, test_file)

        else:
            # prepare for checkpoint
            ckpt_dir = './ckpt/' + FLAGS.input_file + '/'

            ckpt_dir = ckpt_dir + 'E{}_BS{}_S{}_FC{}_L{}/'.format(
                FLAGS.total_epoch, FLAGS.batch_size, FLAGS.seq_length, FLAGS.fc_hidden_unit, FLAGS.lstm_unit)

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            check = ckpt_dir + "*.meta"
            ckpt_idx = len(glob.glob(check))
            ckpt_path = ckpt_dir + "train_result" + str(ckpt_idx+1) + ".ckpt"

            tf.global_variables_initializer().run()
            run_train(sess, train_file)
            saver.save(sess, ckpt_path)
            print('  * Variables are saved: %s *' % ckpt_path)
            run_test(sess, test_file)

if __name__ == '__main__':
    tf.app.run()