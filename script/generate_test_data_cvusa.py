'''
This file is a data loader that output three types of data: polar_sat, sat, grd
'''

import cv2
import random
import numpy as np
import os


class InputData:

    img_root = '../../Data/CVUSA/'


    def __init__(self, data_type='CVUSA'):

        self.data_type = data_type
        self.img_root = '../../Data/' + self.data_type + '/'

        self.train_list = self.img_root + 'splits/train-19zl.csv'
        self.test_list = self.img_root + 'splits/val-19zl.csv'

        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0].replace('bing', 'polar').replace('jpg', 'png'), data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)


        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0].replace('bing', 'polar').replace('jpg', 'png'), data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)


    def next_batch_scan(self, grd_noise=360, FOV=360):

        grd_width = int(FOV/360*512)


        grd_shift = []

        for i in range(self.test_data_size):
            img_idx = self.__cur_test_id + i

            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][2])
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]
            if not os.path.exists('../OrientationUnknwonTestImages/CVUSA/FOV' + str(FOV) + '/'):
                os.makedirs('../OrientationUnknwonTestImages/CVUSA/FOV' + str(FOV) + '/')

            cv2.imwrite('../OrientationUnknwonTestImages/CVUSA/FOV'+str(FOV)+'/' + self.id_test_list[img_idx][0].split('/')[-1].replace('jpg', 'png'),
                        img_dup)

            angle_of_img_center = ((512-random_shift)/512*360+FOV/2)%360 - 180

            orientation_gth = (np.around((angle_of_img_center + 180 -FOV/2)%360/360*64)).astype(np.int)

            grd_shift.append([self.id_test_list[img_idx][0].split('/')[-1].replace('jpg', 'png'), angle_of_img_center, orientation_gth])

        with open('../OrientationUnknwonTestImages/CVUSA/FOV'+str(FOV)+'orien.txt', 'w') as f:
            for imgname, angle, orien_gth in grd_shift:
                f.write(imgname + ' ' + str(angle) + ' ' + str(orien_gth) + '\n')
                # first value: image name
                # second value: the ground truth orientation of the image center
                #               (-180 degree to 180 degree, zero degree corresponds to the north direction,
                #               0-180 degree corresponds to clock-wise rotation)
                # third value: the position at which the maximum value of the correlation
                #              (between ground and polar-transformed aerial features) results is expected to appear.
                #              (within the range of [0, 64), as the feature length of polar transformed aerial image
                #               along the azimuth angle is 64).


data = InputData()

np.random.seed(2019)
data.next_batch_scan(grd_noise=360, FOV=70)



