import cv2
import random
import numpy as np
# load the yaw, pitch angles for the street-view images and yaw angles for the aerial view
import scipy.io as sio
import os

class InputData:
    # the path of your CVACT dataset

    img_root = '../../Data/ANU_data_small/'

    # yaw_pitch_grd = sio.loadmat('./CVACT_orientations/yaw_pitch_grd_CVACT.mat')
    # yaw_sat = sio.loadmat('./CVACT_orientations/yaw_radius_sat_CVACT.mat')

    posDistThr = 25
    posDistSqThr = posDistThr * posDistThr

    panoCropPixels = int(832 / 2)

    panoRows = 128

    panoCols = 512

    satSize = 256

    def __init__(self, polar=1):
        self.polar = polar

        self.allDataList = './OriNet_CVACT/CVACT_orientations/ACT_data.mat'
        print('InputData::__init__: load %s' % self.allDataList)

        self.__cur_allid = 0  # for training
        self.id_alllist = []
        self.id_idx_alllist = []

        # load the mat

        anuData = sio.loadmat(self.allDataList)

        idx = 0
        for i in range(0, len(anuData['panoIds'])):
            grd_id_ori = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_zoom_2.jpg'

            grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'
            grd_id_ori_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][
                i] + '_zoom_2_sem.jpg'
            grd_id_align_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][
                i] + '_zoom_2_aligned_sem.jpg'

            polar_sat_id_ori = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'

            sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.png'
            sat_id_sem = self.img_root + '_' + anuData['panoIds'][i] + '/' + anuData['panoIds'][i] + '_satView_sem.jpg'
            self.id_alllist.append([grd_id_ori, grd_id_align, grd_id_ori_sem, grd_id_align_sem, sat_id_ori, sat_id_sem,
                                    anuData['utm'][i][0], anuData['utm'][i][1], polar_sat_id_ori])
            self.id_idx_alllist.append(idx)
            idx += 1
        self.all_data_size = len(self.id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', self.all_data_size)

        # partion the images into cells

        self.utms_all = np.zeros([2, self.all_data_size], dtype=np.float32)
        for i in range(0, self.all_data_size):
            self.utms_all[0, i] = self.id_alllist[i][6]
            self.utms_all[1, i] = self.id_alllist[i][7]

        self.training_inds = anuData['trainSet']['trainInd'][0][0] - 1

        self.trainNum = len(self.training_inds)

        self.trainList = []
        self.trainIdList = []
        self.trainUTM = np.zeros([2, self.trainNum], dtype=np.float32)
        for k in range(self.trainNum):
            self.trainList.append(self.id_alllist[self.training_inds[k][0]])
            self.trainUTM[:, k] = self.utms_all[:, self.training_inds[k][0]]
            self.trainIdList.append(k)

        self.__cur_id = 0  # for training

        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)

        self.valList = []
        self.valUTM = np.zeros([2, self.valNum], dtype=np.float32)
        for k in range(self.valNum):
            self.valList.append(self.id_alllist[self.val_inds[k][0]])
            self.valUTM[:, k] = self.utms_all[:, self.val_inds[k][0]]
        # cur validation index
        self.__cur_test_id = 0

    def next_batch_scan(self, grd_noise=360, FOV=360):

        grd_width = int(FOV / 360 * 512)

        grd_shift = []

        for i in range(len(self.valList)):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.valList[img_idx][4])
            if img is None or img.shape[0] != img.shape[1]:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][4], i))
                continue

            # polar satellite
            img = cv2.imread(self.valList[img_idx][-1])
            if img is None or img.shape[0] != self.panoRows or img.shape[1] != self.panoCols:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][4], i))
                continue

            # ground
            img = cv2.imread(self.valList[img_idx][1])
            if img is None or img.shape[0] * 4 != img.shape[1]:
                print('InputData::next_pair_batch: read fail: %s, %d, ' % (self.valList[img_idx][2], i))
                continue
            img = cv2.resize(img, (self.panoCols, self.panoRows), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            j = np.arange(0, 512)
            a = np.random.rand()
            random_shift = int(a * 512 * grd_noise / 360)
            img_dup = img[:, ((j - random_shift) % 512)[:grd_width], :]

            if not os.path.exists('../OrientationUnknwonTestImages/CVACT/FOV' + str(FOV) + '/'):
                os.makedirs('../OrientationUnknwonTestImages/CVACT/FOV' + str(FOV) + '/')

            cv2.imwrite(
                '../OrientationUnknwonTestImages/CVACT/FOV' + str(FOV) + '/' + self.valList[img_idx][0].split('/')[
                    -1].replace('jpg', 'png'),
                img_dup)

            angle_of_img_center = ((512 - random_shift) / 512 * 360 + FOV / 2) % 360 - 180

            orientation_gth = (np.around((angle_of_img_center + 180 - FOV / 2) % 360 / 360 * 64)).astype(np.int)

            grd_shift.append([self.valList[img_idx][0].split('/')[-1].replace('jpg', 'png'), angle_of_img_center,
                              orientation_gth])

        with open('../OrientationUnknwonTestImages/CVACT/FOV' + str(FOV) + 'orien.txt', 'w') as f:
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
data.next_batch_scan(grd_noise=360, FOV=90)

