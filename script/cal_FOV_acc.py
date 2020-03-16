#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:12:08 2019

@author: yujiao
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import numpy as np
import scipy.io as scio
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='VGG_13_conv_v2_cir')
parser.add_argument('--data_type', type=str, help='network type', default='CVUSA')

parser.add_argument('--test_grd_noise', type=int, help='0~360', default=360)

parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 100, 120, 180, 360', default=180)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type
data_type = args.data_type

test_grd_noise = args.test_grd_noise

test_grd_FOV = args.test_grd_FOV


def cal_distance(sat_matrix, grd_matrix, pred_orien):
    '''
    :param sat_matrix: shape = [Num_sat, height, sat_width, channel]
    :param grd_matrix: shape = [grd_batch, height, grd_width, channel]
    :param pred_orien: shape = [Num_sat, grd_batch]
    :return:
    '''
    Num_sat, height, sat_width, channel = sat_matrix.shape
    grd_batch, _, grd_width, _ = grd_matrix.shape

    # and crop the corresponding part with ground feature######################
    sat = np.tile(sat_matrix[:, np.newaxis, ...],
                  [1, grd_batch, 1, 1, 1])  # shape = [Num_sat, grd_batch, sat_height, sat_width, sat_channel]
    i = np.arange(Num_sat)
    j = np.arange(grd_batch)
    k = np.arange(sat_width)
    i = i.astype(np.int32)
    j = j.astype(np.int32)
    k = k.astype(np.int32)
    x, y, z = np.meshgrid(i, j, k, indexing='ij')
    index = (z + pred_orien[..., np.newaxis]) % sat_width
    sat = (sat.transpose([0, 1, 3, 2, 4])[x, y, index, :, :]).transpose([0, 1, 3, 2, 4])
    # shape = [batch_sat, batch_grd, sat_height, sat_width, sat_channel]

    sat = sat[:, :, :, 0:grd_width, :].reshape([Num_sat, grd_batch, height * grd_width * channel])

    #######Third step: l2-normalize the cropped satellite feature vector####
    sat_normalize = sat / np.linalg.norm(sat, axis=-1, keepdims=True)

    #######Fourth step: reshape ground and sat feature into vector #######
    grd_vector = np.reshape(grd_matrix[:, :, 0:grd_width, :], [grd_batch, height * grd_width * channel])
    #    sat_vector = np.reshape(sat_normalize, [sat_batch, grd_batch, grd_height*grd_width*grd_channel])

    distance = 2 - 2 * np.einsum('bij, ij->bi', sat_normalize, grd_vector).transpose()
    # shape = [batch_grd, batch_sat]

    return distance


grd_width = int(np.ceil(test_grd_FOV/360*64))

file = '../Result/' + data_type + '/Descriptor/test_grd_noise_' + str(test_grd_noise) + '_test_grd_FOV_' + str(
            test_grd_FOV) + '_' + network_type + '.mat'

data = scio.loadmat(file)
grd_descriptor = data['grd_descriptor'].astype(np.float32)
sat_descriptor = data['sat_descriptor'].astype(np.float32)

sat_descriptor = np.tile(sat_descriptor, [10, 1, 1, 1])

sat_matrix = tf.placeholder(tf.float32, [None, 4, 64, 16], name='sat_x')
grd_matrix = tf.placeholder(tf.float32, [None, 4, grd_width, 16], name='grd_x')

from cir_net_FOV import *
out, orien = corr(sat_matrix, grd_matrix)
#_, distance, orien = corr_crop_distance(sat_matrix, grd_matrix)
# distance.shape = [batch_grd, Num_sat]
# orien.shape = [Num_sat, batch_grd]

data_amount = grd_descriptor.shape[0]
top1_percent = int(data_amount * 0.01) + 1

pred_orientation = np.zeros(data_amount)


#feed_dict = {sat_matrix: sat_descriptor, grd_matrix: grd_descriptor[:500,...]}
#dist_array = sess.run(distance, feed_dict=feed_dict)

sess = tf.Session()
acc = np.zeros(4)
for i in range(400):
    print(i)
    batch_start = int(data_amount * i / 400)
    if i < 399:
        batch_end = int(data_amount * (i + 1) / 400)
    else:
        batch_end = data_amount

    feed_dict = {sat_matrix: sat_descriptor, grd_matrix: grd_descriptor[batch_start: batch_end,...]}
    pred_orien = sess.run(orien, feed_dict=feed_dict)
    dist_array = cal_distance(sat_descriptor, grd_descriptor[batch_start: batch_end,...], pred_orien)

    gt_dist = np.array(
                [dist_array[index, batch_start + index] for index in range(batch_end - batch_start)]).reshape(
                [batch_end - batch_start, 1])

    pred_orientation[batch_start:batch_end] = pred_orien[batch_start:batch_end, :].diagonal()

    prediction = np.sum(dist_array < gt_dist, axis=-1)
    acc[0] += np.sum(prediction < 1)
    acc[1] += np.sum(prediction < 5)
    acc[2] += np.sum(prediction < 10)
    acc[3] += np.sum(prediction < top1_percent)

val_accuracy = acc / data_amount

orientation_gth = data['orientation_gth']

orien_acc = np.sum(orientation_gth==pred_orientation)/data_amount

orien_error = np.abs(orientation_gth - pred_orientation)
orien_acc_5 = np.sum(orien_error<2)/data_amount

scio.savemat(file, {'orientation_gth': orientation_gth, 'pred_orientation': pred_orientation, 
                    'acc': acc, 'orien_acc': orien_acc, 'orien_acc_5': orien_acc_5,
                    'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor})
