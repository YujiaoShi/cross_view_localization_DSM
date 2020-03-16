# import tensorflow as tf

from VGG import VGG16
from VGG_cir import VGG16_cir

# from utils import *
import tensorflow as tf



def tf_shape(x, rank):
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]


def corr(sat_matrix, grd_matrix):

    s_h, s_w, s_c = sat_matrix.get_shape().as_list()[1:]
    g_h, g_w, g_c = grd_matrix.get_shape().as_list()[1:]

    assert s_h == g_h, s_c == g_c

    def warp_pad_columns(x, n):
        out = tf.concat([x, x[:, :, :n, :]], axis=2)
        return out

    n = g_w - 1
    x = warp_pad_columns(sat_matrix, n)
    f = tf.transpose(grd_matrix, [1, 2, 3, 0])
    out = tf.nn.conv2d(x, f,  strides=[1, 1, 1, 1], padding='VALID')
    h, w = out.get_shape().as_list()[1:-1]
    assert h==1, w==s_w

    out = tf.squeeze(out)  # shape = [batch_sat, w, batch_grd]
    orien = tf.argmax(out, axis=1)  # shape = [batch_sat, batch_grd]

    return out, tf.cast(orien, tf.int32)


def crop_sat(sat_matrix, orien, grd_width):
    batch_sat, batch_grd = tf_shape(orien, 2)
    h, w, channel = sat_matrix.get_shape().as_list()[1:]
    sat_matrix = tf.expand_dims(sat_matrix, 1) # shape=[batch_sat, 1, h, w, channel]
    sat_matrix = tf.tile(sat_matrix, [1, batch_grd, 1, 1, 1])
    sat_matrix = tf.transpose(sat_matrix, [0, 1, 3, 2, 4])  # shape = [batch_sat, batch_grd, w, h, channel]

    orien = tf.expand_dims(orien, -1) # shape = [batch_sat, batch_grd, 1]

    i = tf.range(batch_sat)
    j = tf.range(batch_grd)
    k = tf.range(w)
    x, y, z = tf.meshgrid(i, j, k, indexing='ij')

    z_index = tf.mod(z + orien, w)
    x1 = tf.reshape(x, [-1])
    y1 = tf.reshape(y, [-1])
    z1 = tf.reshape(z_index, [-1])
    index = tf.stack([x1, y1, z1], axis=1)

    sat = tf.reshape(tf.gather_nd(sat_matrix, index), [batch_sat, batch_grd, w, h, channel])

    index1 = tf.range(grd_width)
    sat_crop_matrix = tf.transpose(tf.gather(tf.transpose(sat, [2, 0, 1, 3, 4]), index1), [1, 2, 3, 0, 4])
    # shape = [batch_sat, batch_grd, h, grd_width, channel]
    assert sat_crop_matrix.get_shape().as_list()[3] == grd_width

    return sat_crop_matrix


def corr_crop_distance(sat_vgg, grd_vgg):
    corr_out, corr_orien = corr(sat_vgg, grd_vgg)
    sat_cropped = crop_sat(sat_vgg, corr_orien, grd_vgg.get_shape().as_list()[2])
    # shape = [batch_sat, batch_grd, h, grd_width, channel]

    sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])

    distance = 2 - 2 * tf.transpose(tf.reduce_sum(sat_matrix * tf.expand_dims(grd_vgg, axis=0), axis=[2, 3, 4]))
    # shape = [batch_grd, batch_sat]

    return sat_matrix, distance, corr_orien




def VGG_13_conv_v2_cir(x_sat, x_grd, keep_prob, trainable):
    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output
    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)

    sat_matrix, distance, pred_orien = corr_crop_distance(sat_vgg, grd_vgg)

    return sat_vgg, grd_vgg, distance, pred_orien


