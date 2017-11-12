#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 17-10-9 4:23 PM

@author: limengyan
"""
from pandas import *
import numpy as np
import math
import scipy.misc
import cv2
import time
import matplotlib.pyplot as plt


def psnr(img1, img2):
    '''
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    if img1.shape[-1] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        img1 = img1[:, :, 0]

    if img2.shape[-1] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
        img2 = img2[:, :, 0]

    img1 = scipy.misc.imread(img1, mode='YCbCr')
    img2 = scipy.misc.imread(img2, mode='YCbCr')
    img1 = img1[:, :, 0]
    img2 = img2[:, :, 0]

    img1 = scipy.misc.imread(img1, mode='RGB')
    img2 = scipy.misc.imread(img2, mode='RGB')
    '''

    img1 = scipy.misc.imread(img1, flatten=True)
    img2 = scipy.misc.imread(img2, flatten=True)

    imgdff = img1 - img2
    print imgdff.shape
    imgdff = imgdff[:]
    mse = np.mean(imgdff ** 2)

    if mse == 0:
        psnr = -1
    else:
        PIXEL_MAX = 255.0
        psnr = 10 * np.log10(PIXEL_MAX ** 2 / mse)

    # print "MSE: [%.8f], PSNR: [%2.4f] dB" % (mse, psnr)
    return mse, psnr

print psnr('lena.jpg', 'EDSR.png')
print psnr('./samples/evaluate/Set5_eval/2_valid_gen.png',
           './samples/evaluate/Set5_eval/2_valid_hr.png')
"""
total_mse = 0
total_psnr = 0

for i in xrange(0, 14):
    MSE, PSNR = psnr('./samples/evaluate/Set14_eval/%d_valid_gen.png' % i,
                    './samples/evaluate/Set14_eval/%d_valid_hr.png' % i)
    total_mse += MSE
    total_psnr += PSNR

MSE = total_mse / 14
PSNR = total_psnr / 14
print "MSE: [%.8f], PSNR: [%2.4f] dB" % (MSE, PSNR)


training = []

for i in range(10, 2010, 10):
    step_time = time.time()
    MSE, PSNR = psnr('./samples/srgan_gan/train_%d.png' % i,
                    './samples/srgan_gan/_train_sample_384.png')
    print "Epoch [%03d]  MSE: [%.8f], PSNR: [%2.4f] dB, time: [%4.4fs] s" % (
        i, MSE, PSNR, time.time()-step_time)
    training.append([PSNR])

df = DataFrame(training, columns=['PSNR'], index=np.arange(10, 2010, 10))
plt.show(df.plot())
"""



