import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def lstm_fn(height):
    if height == FLAGS.num_lstm_layer-1:
        return tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_unit, state_is_tuple=True,
                                                       reuse = tf.get_variable_scope().reuse)
    else:
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_unit, state_is_tuple=True,
                                                       reuse = tf.get_variable_scope().reuse), output_keep_prob=0.5)

def build():
    """Build the networks
    """
    X = tf.placeholder("float", [None, FLAGS.seq_length, FLAGS.input_dim])
    V = tf.placeholder("float", [None, 1])
    L = tf.placeholder("float", [None, FLAGS.num_class])
    Y = tf.placeholder("float", [None, 1])

    tf.add_to_collection('input', X)
    tf.add_to_collection('input', V)
    tf.add_to_collection('ground_truth', L)
    tf.add_to_collection('ground_truth', Y)

    with tf.variable_scope("lstm_layers") as scope:

        lstm_C = tf.contrib.rnn.MultiRNNCell([lstm_fn(i) for i in range(FLAGS.num_lstm_layer)], state_is_tuple=True)
        lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_C, X, dtype=tf.float32)

        normalization_factor = 4
        concat_hidden_unit = 10

        # concatenate original V with LSTM Output
        V_fc = tf.layers.dense(V / normalization_factor, concat_hidden_unit, activation=tf.nn.relu)
        V_concat = tf.concat([lstm_outputs[:, -1], V_fc], 1)
        lstm_fc = tf.layers.dense(V_concat, FLAGS.fc_hidden_unit, activation=tf.nn.relu)


    with tf.variable_scope("localization") as scope:
        FC1 = tf.layers.dense(lstm_fc, FLAGS.fc_hidden_unit, activation=tf.nn.relu)
        S = tf.layers.dense(FC1, FLAGS.num_class, activation=None)
        S_S = tf.nn.softmax(S)

    with tf.variable_scope("regression") as scope:
        FC2 = tf.layers.dense(lstm_fc, FLAGS.fc_hidden_unit * FLAGS.num_class, activation=tf.nn.relu)
        FC2 = tf.reshape(FC2, [-1, FLAGS.fc_hidden_unit, FLAGS.num_class])
        attention_S = tf.reshape(S_S, [-1, 1, FLAGS.num_class])
        attention_FC2 = tf.multiply(FC2, attention_S)
        attention_FC2 = tf.reshape(attention_FC2, [-1, FLAGS.fc_hidden_unit * FLAGS.num_class])

        FC3 = tf.layers.dense(attention_FC2, FLAGS.fc_hidden_unit, activation=tf.nn.relu)
        O = tf.layers.dense(FC3, 1, activation=None)

    with tf.name_scope("Loss") as scope:
        loss_L = tf.nn.softmax_cross_entropy_with_logits(labels=L, logits=S)
        loss_L = tf.reduce_mean(loss_L)
        loss_R = tf.reduce_mean(tf.square(O - Y))
        loss = loss_L + loss_R

    with tf.name_scope("Train") as scope:
        vars_lstm = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm_layers')
        vars_L = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='localization')
        vars_R = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression')

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss,
                                                                                 var_list=[vars_lstm, vars_L, vars_R])

    for op in [optimizer, loss]:
        tf.add_to_collection('train_ops', op)

    for op in [S_S, O]:
        tf.add_to_collection('test_ops', op)