"""
Author: HaoZhi
Date: 2024-07-09 09:35:14
LastEditors: HaoZhi
LastEditTime: 2024-07-09 14:52:27
Description: 
"""
import os

import psutil
import numpy as np
import nibabel as nib
from scipy import ndimage
from joblib import Parallel, delayed


class DataLoader(object):
    def __init__(self, if_fast, split_margin) -> None:
        if if_fast:
            self.resample = 3.0
        else:
            self.resample = 1.5
        self.if_fast = if_fast
        self.split_margin = split_margin

    def load_by_nib(self, img_path):
        img_in_orig = nib.load(img_path)
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)
        img_in = nib.as_closest_canonical(img_in)
        return img_in_orig, img_in

    def change_sacping(
        self,
        img_in,
        new_spacing=None,
        new_size=None,
        order=0,
        force_affine=None,
        dtype=None,
    ):
        data = img_in.get_fdata()
        img_spacing = np.array(img_in.header.get_zooms())
        if new_size is None:
            if isinstance(new_spacing, float):
                new_spacing = [
                    new_spacing,
                ] * 3
            new_spacing = np.array(new_spacing)
            zoom = img_spacing / new_spacing
        else:
            zoom = np.array(new_size) / np.array(img_in.shape)
            new_spacing = img_spacing / zoom
        new_affine = np.copy(img_in.affine)
        new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
        new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
        new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]
        img_sm = ndimage.zoom(data, zoom, order=order)
        if dtype is not None:
            img_sm = img_sm.astype(dtype)
        if force_affine is not None:
            new_affine = force_affine
        return nib.Nifti1Image(img_sm, new_affine)

    def do_triple_split(self, img):
        if_split = (
            np.prod(img.shape) > 256 * 256 * 900
            and img.shape[2] > 200
            and (not self.if_fast)
        )
        qform = img.get_qform()
        sform = img.get_sform()
        if if_split:
            third = img.shape[2] // 3
            img_d = img.get_fdata()
            img1 = nib.Nifti1Image(img_d[:, :, : third + self.split_margin], img.affine)
            img1.set_qform(qform)
            img1.set_sform(sform)
            img2 = nib.Nifti1Image(
                img_d[
                    :,
                    :,
                    third + 1 - self.split_margin : third * 2 + self.split_margin,
                ],
                img.affine,
            )
            img2.set_qform(qform)
            img2.set_sform(sform)
            img3 = nib.Nifti1Image(
                img_d[:, :, third * 2 + 1 - self.split_margin :], img.affine
            )
            img3.set_qform(qform)
            img3.set_sform(sform)
            img = [img1, img2, img3]
        else:
            img.set_qform(qform)
            img.set_sform(sform)
            img = [img]
            third = 0
        return third, img

    def __call__(self, img_in_path, img_out_path):
        img_in_orig, img_in = self.load_by_nib(img_in_path)
        print(
            f"origin data shape is {img_in_orig.shape}, spacing is {img_in_orig.header.get_zooms()}"
        )
        img_resample = self.change_sacping(
            img_in, new_spacing=self.resample, order=3, dtype=np.int32
        )
        print(
            f"Resample data shape is {img_resample.shape}, spacing is {img_resample.header.get_zooms()}"
        )
        if_split, imgs_list = self.do_triple_split(img_resample)
        if if_split:
            print("Input Data contains too much slices, divide it into three parts")
        tmp_save_folder = os.path.join(img_out_path, "tmp_input")
        if not os.path.exists(tmp_save_folder):
            os.makedirs(tmp_save_folder)
        print("Saving tmp input data...")
        for idx, img in enumerate(imgs_list):
            nib.save(
                img, os.path.join(tmp_save_folder, "s0" + str(idx + 1) + ".nii.gz")
            )
        tmp_pred_seg = nib.Nifti1Image(
            np.zeros(shape=img_resample.shape, dtype="uint8"), img_resample.affine
        )
        return img_in_orig, img_in, tmp_pred_seg, if_split


if __name__ == "__main__":
    img_in = r"D:\data\Abdoman\dcm\055_230904444147\abdoman_1.25.nii.gz"
    img_out = r"D:\data\Abdoman\dcm\055_230904444147\tmp"
    dataloader = DataLoader(if_fast=False)
    dataloader(img_in, img_out)
