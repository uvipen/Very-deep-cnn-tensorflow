"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import tensorflow as tf


class Very_deep_cnn(object):
    def __init__(self, batch_size=128, num_classes=14, depth=9, num_embedding=69, embedding_dim=16, stddev_init=0.01):
        super(Very_deep_cnn, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        if depth == 9:
            self.num_conv_block = [1, 1, 1, 1]
        elif depth == 17:
            self.num_conv_block = [2, 2, 2, 2]
        elif depth == 29:
            self.num_conv_block = [5, 5, 2, 2]
        elif depth == 49:
            self.num_conv_block = [8, 8, 5, 3]
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.stddev_init = stddev_init
        self.base_num_features = 64

    def forward(self, input, is_training=True):
        with tf.variable_scope("embedding"):
            params = tf.Variable(tf.random_uniform([self.num_embedding, self.embedding_dim], -1.0, 1.0),
                                 name="weight")
            embedding = tf.nn.embedding_lookup(params, input)

        with tf.variable_scope("first_layer"):
            weight = self._initialize_weight([3, self.embedding_dim, 64], self.stddev_init)
            bias = self._initialize_bias([64])
            conv = tf.nn.conv1d(value=embedding, filters=weight, stride=1, padding="SAME", name='conv')
            output = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

        for idx, num_conv_block in enumerate(self.num_conv_block):
            output = self._create_conv_block(output, self.base_num_features * pow(2, idx), is_training,
                                             "conv_block_{}".format(idx + 1), num_conv_block, self.stddev_init)

        output = tf.transpose(output, [0, 2, 1])
        output, _ = tf.nn.top_k(output, k=8, name='k-maxpool')
        flatten = tf.reshape(output, (-1, output.get_shape().as_list()[1] * output.get_shape().as_list()[2]))

        output = self._create_fc(flatten, 2048, "fc1", self.stddev_init)
        output = self._create_fc(output, 2048, "fc2", self.stddev_init)
        output = self._create_fc(output, self.num_classes, "fc3", self.stddev_init)

        return output

    def _create_conv_block(self, input, out_channels, is_training, variable_scope, num_blocks, stddev=0.01):
        output = input
        with tf.variable_scope(variable_scope):
            for idx in range(num_blocks):
                with tf.variable_scope("block_{}".format(idx + 1)):
                    with tf.variable_scope("lower_part"):
                        weight = self._initialize_weight([3, output.get_shape().as_list()[2], out_channels],
                                                         stddev)
                        bias = self._initialize_bias([out_channels])
                        conv = tf.nn.conv1d(value=output,
                                            filters=weight, stride=1, padding="SAME",
                                            name='conv')
                        batchnorm = tf.layers.batch_normalization(conv, training=is_training, name="batchnorm")
                        output = tf.nn.relu(tf.nn.bias_add(batchnorm, bias), name="relu")

                    with tf.variable_scope("upper_part"):
                        weight = self._initialize_weight([3, out_channels, out_channels],
                                                         stddev)
                        bias = self._initialize_bias([out_channels])
                        conv = tf.nn.conv1d(value=output,
                                            filters=weight, stride=1, padding="SAME",
                                            name='conv')
                        batchnorm = tf.layers.batch_normalization(conv, training=is_training, name="batchnorm")
                        output = tf.nn.relu(tf.nn.bias_add(batchnorm, bias), name="relu")
            # return tf.nn.max_pool(value=output, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1],
            #                       padding="SAME", name='maxpool')
            return tf.layers.max_pooling1d(inputs=output, pool_size=3, strides=2, padding='same', name="maxpool")

    def _create_fc(self, input, out_channels, variable_scope, stddev=0.01):
        with tf.variable_scope(variable_scope):
            weight = self._initialize_weight([input.get_shape().as_list()[1], out_channels], stddev)
            bias = self._initialize_bias([out_channels])
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(input, weight), bias, name="dense"))
            # return tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(input, weight), bias, name="dense")), 0.5)

    def _initialize_weight(self, shape, stddev):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32, name='weight'))

    def _initialize_bias(self, shape):
        return tf.Variable(tf.constant(0, shape=shape, dtype=tf.float32, name='bias'))

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def accuracy(self, logits, labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), dtype=tf.float32))

    def confusion_matrix(self, logits, labels):
        return tf.confusion_matrix(tf.cast(labels, tf.int64), tf.argmax(logits, 1), num_classes=self.num_classes)
