# -*- coding: utf-8 -*-
import numpy as np
import pydicom as pydicom
import os
import SimpleITK as itk
from PIL import Image, ImageDraw
from queue import Queue
import gc
import copy
import cv2
import nipy
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border, mark_boundaries

# 计算dice coefficient
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

# 读取mhd文件
def read_mhd_image(file_path, rejust=False):
    header = itk.ReadImage(file_path)
    image = np.array(itk.GetArrayFromImage(header))
    if rejust:
        image[image < -70] = -70
        image[image > 180] = 180
        image = image + 70
    return np.array(image)

# 保存mhd文件
def save_mhd_image(image, file_name):
    header = itk.GetImageFromArray(image)
    itk.WriteImage(header, file_name)

#图像对比可视化
def original_gt_seg_diff_vis(test_image, test_mask, test_seg):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    ax1.imshow(test_image, cmap='bone')
    ax1.set_title('CT Slice')
    ax2.imshow(test_mask)
    ax2.set_title('Ground Truth')
    ax3.imshow(test_seg)
    ax3.set_title('')
    ax4.imshow((test_seg > 0) ^ (test_mask > 0), cmap='gist_earth')
    ax4.set_title('Difference')
    plt.show(fig)

def make_mb_image(i_img, i_gt, i_pred, ds_op=lambda x: x[::4,::4]):
    n_img = (i_img-i_img.mean())/(2*i_img.std())+0.5
    c_img = plt.cm.bone(n_img)[:, :, :3]
    c_img = mark_boundaries(ds_op(c_img), label_img=ds_op(i_pred), color = (1, 0, 0), mode='thick')
    c_img = mark_boundaries(c_img, label_img=ds_op(i_gt), color=(0, 1, 0), mode='thick|')
    return c_img