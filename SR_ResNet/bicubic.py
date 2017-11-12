#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 17-10-11 10:20 AM

@author: limengyan
"""
import numpy as np
from PIL import Image
import cv2
import scipy.misc
import scipy.ndimage
import skimage.transform
import tensorflow as tf
import tensorlayer as tl
from utils import *


def bicubic(img, scale):

    print "Scipy_misc Bicubic RGB..."
    img = scipy.misc.imread(img, mode='RGB')
    h, w, _ = img.shape
    img = scipy.misc.imresize(img, size=[h/scale, w/scale], interp='bicubic', mode=None)
    return img


for i in xrange(1, 15):
    valid_hr_img = './data/test/Set14/img_%03d.png' % i
    img = bicubic(valid_hr_img, 4)
    scipy.misc.imsave('./data/test/Set14_lr/img_%03d_lr.png' % i, img)
