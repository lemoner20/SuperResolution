#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 17-10-19 8:55 PM

@author: limengyan
"""
import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.

    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)


    ###========================== DEFINE MODEL ============================###
    for i in xrange(14, 15):

        imid = i  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]

        valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]

        size = valid_lr_img.shape
        t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')

        net_g = SR(t_image, is_train=False, reuse=False)

        ###========================== RESTORE G =============================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir+'/%d_valid_gen.png' % i)
        tl.vis.save_image(valid_lr_img, save_dir+'/%d_valid_lr.png' % i)
        tl.vis.save_image(valid_hr_img, save_dir+'/%d_valid_hr.png' % i)

        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir+'/%d_valid_bicubic.png' % i)