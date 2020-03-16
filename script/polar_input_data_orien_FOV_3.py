
import cv2
import random
import numpy as np


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
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)


    def next_batch_scan(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None, None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        grd_width = int(FOV/360*512)

        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype = np.float32)
        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        grd_shift = np.zeros([batch_size], dtype=np.int)

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
#            print(self.id_test_list[img_idx][0])

            # satellite polar
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][0])
            # img = cv2.resize(img, (616, 112), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat_polar[i, :, :, :] = img


            # satellite
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][1])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            img = cv2.imread(self.img_root + self.id_test_list[img_idx][2])
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            # img -= 100.0
            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img_dup

            grd_shift[i] = random_shift

        self.__cur_test_id += batch_size
#        print(grd_shift[0])

        return batch_sat_polar, batch_sat, batch_grd, (np.around(((512-grd_shift)/512*64)%64)).astype(np.int)



    def next_pair_batch(self, batch_size, grd_noise=360, FOV=360):
        if self.__cur_id == 0:
            for i in range(20):
                random.shuffle(self.id_idx_list)

        if self.__cur_id + batch_size + 2 >= self.data_size:
            self.__cur_id = 0
            return None, None, None, None

        grd_width = int(FOV/360*512)

        batch_sat_polar = np.zeros([batch_size, 128, 512, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 256, 256, 3], dtype=np.float32)
        batch_grd = np.zeros([batch_size, 128, grd_width, 3], dtype=np.float32)
        grd_shift = np.zeros([batch_size,], dtype=np.int)
        i = 0
        batch_idx = 0
        while True:
            if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                break

            img_idx = self.id_idx_list[self.__cur_id + i]
            i += 1

            # satellite polar
            img = cv2.imread(self.img_root + self.id_list[img_idx][0])
            if img is None or img.shape[0] != 128 or img.shape[1] != 512:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i), img.shape)
                continue
            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6    # Red
            batch_sat_polar[batch_idx, :, :, :] = img

            # satellite
            img = cv2.imread(self.img_root + self.id_list[img_idx][1])
            if img is None or img.shape[0] != 750 or img.shape[1] != 750:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][0], i),
                      img.shape)
                continue
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[batch_idx, :, :, :] = img

            # ground
            img = cv2.imread(self.img_root + self.id_list[img_idx][2])
            if img is None or img.shape[0] != 224 or img.shape[1] != 1232:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.img_root + self.id_list[img_idx][1], i), img.shape)
                continue
            img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            random_shift = int(np.random.rand() * 512 * grd_noise / 360)
            img_dup = img[:, ((j-random_shift)%512)[:grd_width], :]

            # img -= 100.0
            img_dup[:, :, 0] -= 103.939  # Blue
            img_dup[:, :, 1] -= 116.779  # Green
            img_dup[:, :, 2] -= 123.6  # Red
            batch_grd[batch_idx, :, :, :] = img_dup
            grd_shift[batch_idx] = random_shift

            batch_idx += 1

        self.__cur_id += i

        return batch_sat_polar, batch_sat, batch_grd, (np.around(((512-grd_shift)/512*64)%64)).astype(np.int)


    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0
