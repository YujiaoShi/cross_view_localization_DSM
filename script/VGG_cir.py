import tensorflow as tf

'''VGG modle with circular convolution'''


class VGG16_cir(object):

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
            # layer 14: max pooling
#            self.layer14_output = self.maxpool_layer(self.layer13_output, 'layer14_maxpool2x2')
#
#            # layer 15: conv3-512
#            self.layer15_output = self.conv_layer(self.layer14_output, 3, 512, 512, trainable, True, 'conv5_1')
#            self.layer15_output = tf.nn.dropout(self.layer15_output, keep_prob, name='conv5_1_dropout')
#            # layer 16: conv3-512
#            self.layer16_output = self.conv_layer(self.layer15_output, 3, 512, 512, trainable, True, 'conv5_2')
#            self.layer16_output = tf.nn.dropout(self.layer16_output, keep_prob, name='conv5_2_dropout')
#            # layer 17: conv3-512
#            self.layer17_output = self.conv_layer(self.layer16_output, 3, 512, 512, trainable, True, 'conv5_3')   # shape = [7, 39]
#            self.layer17_output = tf.nn.dropout(self.layer17_output, keep_prob, name='conv5_3_dropout')

            # self.layer18_output = self.maxpool_layer(self.layer17_output, 'layer18_maxpool2x2')


    def conv2d(self, x, W):
        # w_pad = tf.pad
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                            padding='VALID')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def warp_pad_columns(self, x, n=1):

        out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)
        return tf.pad(out, [[0, 0], [n, n], [0, 0], [0, 0]])

    ############################ layers ###############################
    def conv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        n = int((kernel_dim - 1) / 2)
        x = self.warp_pad_columns(x, n)

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

    def maxpool_layer(self, x, name):
        with tf.name_scope(name):
            maxpool = self.max_pool_2x2(x)
            return maxpool

    def conv_layer2(self, x, kernel_dim, strides, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        '''
        :param x:
        :param kernel_dim: scalar
        :param strides: [1, stride_height, stride_width, 1]
        :param input_dim:
        :param output_dim:
        :param trainable:
        :param activated:
        :param name:
        :param activation_function:
        :return:
        '''
        n = int((kernel_dim - 1) / 2)
        x = self.warp_pad_columns(x, n)

        with tf.variable_scope(name): # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            conv = tf.nn.conv2d(x, weight, strides, padding='VALID')

            if activated:
                out = activation_function(conv + bias)
            else:
                out = conv + bias

            return out

    def conv2(self, x, scope_name, dimension=16):
        with tf.variable_scope(scope_name):
            layer15_output = self.conv_layer2(x, 3, [1, 2, 1, 1], 512, 256, self.trainable, True, 'conv_dim_reduct_1')

            layer16_output = self.conv_layer2(layer15_output, 3, [1, 2, 1, 1], 256, 64, self.trainable, True, 'conv5_dim_reduct_2')

            layer17_output = self.conv_layer(layer16_output, 3, 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [7, 39]

        return layer17_output


