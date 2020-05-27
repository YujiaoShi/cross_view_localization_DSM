import tensorflow as tf


class VGG16(object):

    def __init__(self, x, keep_prob, trainable, name):
        self.trainable = trainable
        self.name = name

        with tf.variable_scope(name):
            # layer 1: conv3-64
            self.layer1_output = self.conv_layer(x, 3, 3, 64, False, True, 'conv1_1')
            # layer 2: conv3-64
            self.layer2_output = self.conv_layer(self.layer1_output, 3, 64, 64, False, True, 'conv1_2')
            # layer3: max pooling
            self.layer3_output = self.maxpool_layer(self.layer2_output, 'layer3_maxpool2x2')

            # layer 4: conv3-128
            self.layer4_output = self.conv_layer(self.layer3_output, 3, 64, 128, False, True, 'conv2_1')
            # layer 5: conv3-128
            self.layer5_output = self.conv_layer(self.layer4_output, 3, 128, 128, False, True, 'conv2_2')
            # layer 6: max pooling
            self.layer6_output = self.maxpool_layer(self.layer5_output, 'layer6_maxpool2x2')

            # layer 7: conv3-256
            self.layer7_output = self.conv_layer(self.layer6_output, 3, 128, 256, False, True, 'conv3_1')
            # layer 8: conv3-256
            self.layer8_output = self.conv_layer(self.layer7_output, 3, 256, 256, False, True, 'conv3_2')
            # layer 9: conv3-256
            self.layer9_output = self.conv_layer(self.layer8_output, 3, 256, 256, False, True, 'conv3_3')  # shape = [28, 154]
            # layer 10: max pooling
            self.layer10_output = self.maxpool_layer(self.layer9_output, 'layer10_maxpool2x2')

            # layer 11: conv3-512
            self.layer11_output = self.conv_layer(self.layer10_output, 3, 256, 512, trainable, True, 'conv4_1')
            self.layer11_output = tf.nn.dropout(self.layer11_output, keep_prob, name='conv4_1_dropout')
            # layer 12: conv3-512
            self.layer12_output = self.conv_layer(self.layer11_output, 3, 512, 512, trainable, True, 'conv4_2')
            self.layer12_output = tf.nn.dropout(self.layer12_output, keep_prob, name='conv4_2_dropout')
            # layer 13: conv3-512
            self.layer13_output = self.conv_layer(self.layer12_output, 3, 512, 512, trainable, True, 'conv4_3')  # shape = [14, 77]
            self.layer13_output = tf.nn.dropout(self.layer13_output, keep_prob, name='conv4_3_dropout')


    def conv2d(self, x, W, strides=[1,1,1,1]):
        # w_pad = tf.pad
        return tf.nn.conv2d(x, W, strides,
                            padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    ############################ layers ###############################
    def conv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name): # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.conv2d(x, weight) + bias)
            else:
                out = self.conv2d(x, weight) + bias

            return out


    def conv_layer2(self, x, kernel_dim, strides, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name): # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.conv2d(x, weight, strides) + bias)
            else:
                out = self.conv2d(x, weight, strides) + bias

            return out


    def maxpool_layer(self, x, name):
        with tf.name_scope(name):
            maxpool = self.max_pool_2x2(x)
            return maxpool


    def conv2(self, x, scope_name, dimension=16):
        with tf.variable_scope(scope_name):
            layer15_output = self.conv_layer2(x, 3, [1, 2, 1, 1], 512, 256, self.trainable, True, 'conv_dim_reduct_1')

            layer16_output = self.conv_layer2(layer15_output, 3, [1, 2, 1, 1], 256, 64, self.trainable, True, 'conv5_dim_reduct_2')

            layer17_output = self.conv_layer(layer16_output, 3, 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [7, 39]

        return layer17_output


