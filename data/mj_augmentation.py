# Code for data augmentation, applied to gait sequences
# (c) MJMJ/2020

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'April 2020'

import os.path as osp
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def mj_mirrorsequence(sample, isof=True, copy=True):
    """
    Returns a new variable (previously copied), not in-place!
    :rtype: numpy.array
    :param sample:
    :param isof: boolean. If True, sign of x-channel is changed (i.e. direction changes)
    :return: mirror sample
    """
    # Make a copy
    if copy:
        newsample = np.copy(sample)
    else:
        newsample = sample

    nt = newsample.shape[0]
    for i in range(nt):
        newsample[i,] = np.fliplr(newsample[i,])
        if i % 2 == 0:
            newsample[i,] = -newsample[i,]

    return newsample


def mj_transformsequence(sample, img_gen, transformation):
    sample_out = np.zeros_like(sample)
    #min_v, max_v = (sample.min(), sample.max())
    abs_max = np.abs(sample).max()
    for i in range(sample.shape[0]):
        I = np.copy(sample[i, ])
        I = np.expand_dims(I, axis=2)
        It = img_gen.apply_transform(I, transformation)

        sample_out[i, ] = It[:, :, 0]

    # Fix range if needed
    if np.abs(sample_out).max() > 3*abs_max: # This has to be normalized
        sample_out = (sample_out /255.0) - 0.5

    return sample_out


def mj_transgenerator(displace=[-5, -3, 0, 3, 5], isof=True):

    if isof:
        ch_sh_range = 0
        br_range = None
    else:
        ch_sh_range = 0.025
        br_range = [0.95, 1.05]

    img_gen = ImageDataGenerator(width_shift_range=displace, height_shift_range=displace,
                                 brightness_range=br_range, zoom_range=0.04,
                                 channel_shift_range=ch_sh_range, horizontal_flip=False)

    return img_gen

# ====================================== MAIN =========================


if __name__ == "__main__":

    #M = np.array([[1,2], [3,4], [5,6]])
    np.random.seed(123)
    import deepdish as dd

    # M = np.random.random((2, 3, 3))
    if False:
        #filepath = "/home/mjmarin/databases/tumgaid/h5/tfimdb_tum_gaid_N150_train_of25_60x60/p001-n01-01.h5"
        filepath = "/home/mjmarin/databases/tumgaid/h5/tfimdb_tum_gaid_N155_test_b01-02_of25_60x60/p005-b02-04.h5"
        D = dd.io.load(filepath)
        M = D["data"]
        M = np.moveaxis(M, 2, 0)
    else:
        filepath = "/home/mjmarin/databases/tumgaid/h5/tfimdb_tum_gaid_N150_train_depth25_60x60/p001-b02-05.h5"
        isOF = False
        # filepath = "/home/mjmarin/databases/tumgaid/dd_files/matimdbtum_gaid_N150_of25_60x60_lite/0_sample000000.h5"
        # isOF = True
        D = dd.io.load(filepath)
        M = D["data"]
        # filepath = "/home/mjmarin/databases/casiab/h5/tfimdb_casia_b_N074_train_of25_60x60/001-nm-04-036-08.h5"
        # D = dd.io.load(filepath)
        # M = D["data"]
        M = np.moveaxis(M, 2, 0)


    M2 = mj_mirrorsequence(M, True, True)

    # print(M)
    # print(M2)

    # Transformations
    # img_gen = ImageDataGenerator(width_shift_range=[-4, 0, 4], height_shift_range=[-4, 0, 4],
    #                              brightness_range=None, zoom_range=0.04,
    #                              channel_shift_range=0, horizontal_flip=False)

    img_gen = mj_transgenerator(isof=isOF)
    for i in range(10):
        transformation = img_gen.get_random_transform((60, 60))
        print(transformation)

    #sample = -0.5*np.ones((50, 60, 60))
    sample = M2

    sample_out = mj_transformsequence(sample, img_gen, transformation)

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.imshow(M[24,])
    plt.figure(1)
    plt.imshow(M2[24,])
    plt.figure(2)
    plt.imshow(sample_out[24,])
    plt.colorbar()
    plt.show()


    print("Done!")

