# CVUSA VGG_gp the input aerial images are under polar transformed according its origin
# We just want to demonstrate the effectiveness of polar transformation

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cir_net_FOV import *
from polar_input_data_orien_FOV_3 import InputData
from distance import *

import tensorflow as tf
import numpy as np

import argparse
import scipy.io as scio
from numpy import fft

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='VGG_13_conv_v2_cir')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=25)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=100)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 100, 120, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 100, 120, 180, 360', default=360)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type

start_epoch = args.start_epoch
polar = args.polar

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

number_of_epoch = args.number_of_epoch

data_type = 'CVUSA'

loss_type = 'l1'

batch_size = 32
is_training = False
loss_weight = 10.0
# number_of_epoch = 100

learning_rate_val = 1e-5
keep_prob_val = 0.8

dimension = 4


# -------------------------------------------------------- #


def validate(dist_array, topK):
    accuracy = 0.0
    data_amount = 0.0

    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[i, :] < gt_dist)
        if prediction < topK:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(dist_array):

    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        pos_dist = tf.diag_part(dist_array)

        pair_n = batch_size * (batch_size - 1.0)

        # satellite to ground
        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

        # ground to satellite
        triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
        loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0

    return loss


if __name__ == '__main__':

    tf.reset_default_graph()

    # import data
    input_data = InputData()

    width = int(test_grd_FOV / 360 * 512)

    # define placeholders
    grd_x = tf.placeholder(tf.float32, [None, 128, width, 3], name='grd_x')
    sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')
    polar_sat_x = tf.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')

    grd_orien = tf.placeholder(tf.int32, [None], name='grd_orien')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    sat_matrix, grd_matrix, distance, pred_orien = VGG_13_conv_v2_cir(polar_sat_x, grd_x, keep_prob, is_training)

    loss = compute_loss(distance)

    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])
    pred_orientation = np.zeros([input_data.get_test_dataset_size()])

    print('setting saver...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    print('setting saver done...')

    global_vars = tf.global_variables()

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.global_variables_initializer())

        print('load model...')

        load_model_path = '../Model/polar_' + str(polar) + '/' + data_type + '/' + network_type \
                          + '/train_grd_noise_' + str(train_grd_noise) + '/train_grd_FOV_' + str(train_grd_FOV) \
                          + '/model.ckpt'
        saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # ---------------------- validation ----------------------

        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()

        np.random.seed(2019)
        val_i = 0
        while True:
            # print('      progress %d' % val_i)
            batch_sat_polar, batch_sat, batch_grd, batch_orien = input_data.next_batch_scan(batch_size, grd_noise=test_grd_noise,
                                                                           FOV=test_grd_FOV)
            if batch_sat is None:
                break

            feed_dict = {polar_sat_x: batch_sat_polar, grd_x: batch_grd, keep_prob: 1.0}
            sat_matrix_val, grd_matrix_val = \
                sess.run([sat_matrix, grd_matrix], feed_dict=feed_dict)

            sat_global_matrix[val_i: val_i + sat_matrix_val.shape[0], :] = sat_matrix_val
            grd_global_matrix[val_i: val_i + grd_matrix_val.shape[0], :] = grd_matrix_val
            orientation_gth[val_i: val_i + grd_matrix_val.shape[0]] = batch_orien
            val_i += sat_matrix_val.shape[0]

        file = '../Result/CVUSA/Descriptor/' \
               + 'train_grd_noise_' + str(train_grd_noise) + '_train_grd_FOV_' + str(train_grd_FOV) \
               + 'test_grd_noise_' + str(test_grd_noise) + '_test_grd_FOV_' + str(test_grd_FOV) \
               + '_' + network_type + '.mat'
        scio.savemat(file, {'orientation_gth': orientation_gth,
                            'grd_descriptor': grd_global_matrix, 'sat_descriptor': sat_global_matrix})
        grd_descriptor = grd_global_matrix
        sat_descriptor = sat_global_matrix

        data_amount = grd_descriptor.shape[0]
        top1_percent = int(data_amount * 0.01) + 1

        if test_grd_noise==0:
            sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
            sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)

            grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel])

            dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
            gt_dist = dist_array.diagonal()
            prediction = np.sum(dist_array < gt_dist.reshape(-1, 1), axis=-1)
            loc_acc = np.sum(prediction.reshape(-1, 1) < np.arange(top1_percent), axis=0) / data_amount

            scio.savemat(file, {'loc_acc': loc_acc,
                                'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor})

        else:

            sat_fft = fft.fft(sat_descriptor.transpose([0, 3, 1, 2]))[:, np.newaxis, ...]

            loc_acc = np.zeros(top1_percent)
            for i in range(100):
                print(i)
                batch_start = int(data_amount * i / 100)
                if i < 99:
                    batch_end = int(data_amount * (i + 1) / 100)
                else:
                    batch_end = data_amount

                dist_array, pred_orien = corr_distance_FOV_np(grd_descriptor[batch_start: batch_end, :], sat_descriptor, sat_fft)
                gt_dist = np.array(
                    [dist_array[index, batch_start + index] for index in range(batch_end - batch_start)]).reshape(
                    [batch_end - batch_start, 1])
                pred_orientation[batch_start:batch_end] = pred_orien[batch_start:batch_end, :].diagonal()
                prediction = np.sum(dist_array < gt_dist, axis=-1)

                loc_acc += np.sum(prediction.reshape(-1, 1) < np.arange(top1_percent).reshape(1, -1), axis=0)

            loc_acc = loc_acc / data_amount

            print(loc_acc)

            scio.savemat(file, {'orientation_gth': orientation_gth, 'pred_orientation': pred_orientation,
                                'loc_acc': loc_acc,
                                'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor})


