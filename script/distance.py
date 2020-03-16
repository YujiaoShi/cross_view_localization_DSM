
import numpy as np
from numpy import fft

def corr_distance_FOV_np(grd_matrix, sat_matrix, sat_fft):
    '''
    correlation distance, used in the test process
    :param grd_matrix: shape = [batch_grd, height, grd_width, channel]
    :param sat_matrix: shape = [batch_sat, height, sat_width, channel]
    :param orien: shape = [batch_grd]
    :return:
    '''
    grd_batch, grd_height, grd_width, grd_channel = grd_matrix.shape

    sat_batch, sat_height, sat_width, sat_channel = sat_matrix.shape

    assert grd_height == sat_height, grd_channel == sat_channel
    assert grd_width <= sat_width

    ######################First step: perform correlation and get the orientation shift.################################
    grd_matrix = np.pad(grd_matrix, [[0,0], [0,0], [0,sat_width-grd_width], [0,0]], mode='constant')

    grd_fft = fft.fft(grd_matrix.transpose([0,3,1,2]))[np.newaxis, ...]

    fc = np.sum(sat_fft.conj()*grd_fft, axis=(2,3))
    ifft = fft.fft(fc).real/sat_width  # shape = [batch_sat, batch_grd, sat_width]

    pred_orien = np.argmax(ifft, axis=-1) # shape = [batch_sat, batch_grd]

    #########Second step: shift the satellite feature according to the prediced orientation shift
    # and crop the corresponding part with ground feature######################
    sat_matrix = np.tile(sat_matrix[:, np.newaxis, ...], [1, grd_batch, 1, 1, 1]) # shape = [sat_batch, grd_batch, sat_height, sat_width, sat_channel]
    i = np.arange(sat_batch)
    j = np.arange(grd_batch)
    k = np.arange(sat_width)
    x, y, z = np.meshgrid(i, j, k, indexing='ij')
    index = (z + pred_orien[..., np.newaxis]) % sat_width
    sat_matrix = (sat_matrix.transpose([0, 1, 3, 2, 4])[x, y, index, :, :]).transpose([0, 1, 3, 2, 4])
    # shape = [batch_sat, batch_grd, sat_height, sat_width, sat_channel]

    sat_matrix = sat_matrix[:, :, :, 0:grd_width, :].reshape([sat_batch, grd_batch, grd_height*grd_width*grd_channel])

    #######Third step: l2-normalize the cropped satellite feature vector####
    sat_matrix = sat_matrix/np.linalg.norm(sat_matrix, axis=-1, keepdims=True)

    #######Fourth step: reshape ground and sat feature into vector #######
    grd_vector = np.reshape(grd_matrix[:,:,0:grd_width,:], [grd_batch, grd_height*grd_width*grd_channel])

    distance = 2 - 2 * np.einsum('bij, ij->bi', sat_matrix, grd_vector).transpose()
    # shape = [batch_grd, batch_sat]

    return distance, pred_orien


