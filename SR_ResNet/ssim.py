#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 17-10-21 2:33 PM

@author: limengyan
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.misc


def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    mssim = np.mean(ssim_map)

    return mssim


def ssim(img1, img2):
    img1 = scipy.misc.imread(img1, flatten=True)
    img2 = scipy.misc.imread(img2, flatten=True)

    ssim_value = ssim_exact(img1/255., img2/255.)
    print ssim_value

    return ssim_value

ssim('0869.png', '0869x4.png')