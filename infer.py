"""
Author: HaoZhi
Date: 2024-07-09 10:08:01
LastEditors: HaoZhi
LastEditTime: 2024-07-09 14:41:04
Description: 
"""
"""
Author: HaoZhi
Date: 2024-07-09 10:08:01
LastEditors: HaoZhi
LastEditTime: 2024-07-09 10:16:47
Description: 
"""
import os
import glob
import shutil
from tqdm import tqdm

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

from load_data import DataLoader
from build_model import BuildModel
from nnunet_predict import Predictor
from class_map import class_map
from totalsegmentator.libs import nostdout


class Infer(object):
    def __init__(
        self,
        nii_input_path,
        model_folder,
        tmp_folder,
        nii_out_path,
        if_fast,
        split_margin,
        all_in_gpu,
        mix_precision,
    ) -> None:
        self.nii_input_path = nii_input_path
        self.model_folder = model_folder
        self.tmp_folder = tmp_folder
        self.nii_out_path = nii_out_path
        self.if_fast = if_fast
        self.split_margin = split_margin
        self.all_in_gpu = all_in_gpu
        self.mix_precision = mix_precision
        self.task_id = [256] if self.if_fast else [251, 252, 253, 254, 255]
        self.build_dataloader()
        self.build_model()
        self.build_predictor()
        self.process_class_map()

    def build_dataloader(
        self,
    ):
        self.dataloader = DataLoader(self.if_fast, self.split_margin)

    def build_model(
        self,
    ):
        print("------Build Model------")
        self.model_dict = {}
        model_builder = BuildModel()
        for task_id in self.task_id:
            # print(
            #     "debug: ",
            #     os.path.join(
            #         self.model_folder,
            #         "Task" + str(task_id) + "_TotalSegmentator*",
            #         "*",
            #     ),
            # )
            print(f"Build Task{task_id} Model")
            model_path = glob.glob(
                os.path.join(
                    self.model_folder,
                    "Task" + str(task_id) + "_TotalSegmentator*",
                    "*",
                )
            )[0]

            trainer, params = model_builder(model_path, fp16=True)
            self.model_dict[task_id] = dict(trainer=trainer, params=params)

    def build_predictor(
        self,
    ):
        self.predictor = Predictor()

    def init_tmp_folder(
        self,
    ):
        tmp_input_folder = os.path.join(self.tmp_folder, "tmp_input")
        if not os.path.exists(tmp_input_folder):
            os.makedirs(tmp_input_folder)

        tmp_pred_folder = os.path.join(self.tmp_folder, "tmp_pred")
        if not os.path.exists(tmp_pred_folder):
            os.makedirs(tmp_pred_folder)

        for task_id in self.task_id:
            tmp_pred_folder_task = os.path.join(tmp_pred_folder, "Task" + str(task_id))
            if not os.path.exists(tmp_pred_folder_task):
                os.makedirs(tmp_pred_folder_task)

    def clean_tmp_folder(
        self,
    ):
        tmp_input_folder = os.path.join(self.tmp_folder, "tmp_input")
        if os.path.exists(tmp_input_folder):
            shutil.rmtree(tmp_input_folder)
        tmp_pred_folder = os.path.join(self.tmp_folder, "tmp_pred")
        if os.path.exists(tmp_pred_folder):
            shutil.rmtree(tmp_pred_folder)

    def process_class_map(
        self,
    ):
        self.class_map_inv = {v: k for k, v in class_map["total"].items()}

    def undo_canonical(self, img_can, img_orig):
        img_ornt = io_orientation(img_orig.affine)
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)

        return img_can.as_reoriented(from_canonical)

    def __call__(
        self,
    ):
        print("------Init TMP Folder------")
        self.init_tmp_folder()
        print("------Load Data------")
        img_in_ori, img_in, tmp_pred_seg, split_flag = self.dataloader(
            self.nii_input_path, self.tmp_folder
        )
        tmp_pred_seg_array = tmp_pred_seg.get_fdata()
        self.files_input = glob.glob(
            os.path.join(self.tmp_folder, "tmp_input", "*.nii.gz")
        )
        self.files_input = sorted(
            self.files_input,
            key=lambda x: int(x.split(os.sep)[-1].split("s")[-1].split(".nii.gz")[0]),
        )
        self.files_input = list(map(lambda x: [x], self.files_input))
        print("------Model Predict------")
        for task_id in self.task_id:
            print(f"Predict Task{task_id}...")
            files_outpath = list(
                map(
                    lambda x: x[0].replace("tmp_input", "tmp_pred/Task" + str(task_id)),
                    self.files_input,
                )
            )
            trainer = self.model_dict[task_id]["trainer"]
            params = self.model_dict[task_id]["params"]
            with nostdout(True):
                self.predictor.predict(
                    trainer=trainer,
                    params=params,
                    inputs_list=self.files_input,
                    output_list=files_outpath,
                    num_threads_nifti_save=1,
                    num_threads_preprocessing=1,
                    all_in_gpu=self.all_in_gpu,
                    mixed_precision=self.mix_precision,
                )
            print(f"PostProcess")
            if len(self.task_id) > 1:
                sub_class_map = class_map["Task" + str(task_id)]
            if split_flag > 0:
                print("Merge Split Part")
                part1 = nib.load(files_outpath[0]).get_fdata()[
                    :, :, : -self.dataloader.split_margin
                ]
                if len(self.task_id) > 1:
                    for k, v in sub_class_map.items():
                        tmp_pred_seg_array[:, :, :split_flag][
                            part1 == k
                        ] = self.class_map_inv[v]
                else:
                    tmp_pred_seg_array[:, :, :split_flag] = part1
                part2 = nib.load(files_outpath[1]).get_fdata()[
                    :,
                    :,
                    self.dataloader.split_margin - 1 : -self.dataloader.split_margin,
                ]
                if len(self.task_id) > 1:
                    for k, v in sub_class_map.items():
                        tmp_pred_seg_array[:, :, split_flag : split_flag * 2][
                            part2 == k
                        ] = self.class_map_inv[v]
                else:
                    tmp_pred_seg_array[:, :, split_flag : split_flag * 2] = part2
                part3 = nib.load(files_outpath[2]).get_fdata()[
                    :, :, self.dataloader.split_margin - 1 :
                ]
                if len(self.task_id) > 1:
                    for k, v in sub_class_map.items():
                        tmp_pred_seg_array[:, :, split_flag * 2 :][
                            part3 == k
                        ] = self.class_map_inv[v]
                else:
                    tmp_pred_seg_array[:, :, split_flag * 2 :] = part3
            else:
                # print("Copy Pred Seg...")
                part = nib.load(files_outpath[0]).get_fdata()[:]
                if len(self.task_id) > 1:
                    for k, v in sub_class_map.items():
                        tmp_pred_seg_array[part == k] = self.class_map_inv[v]
                else:
                    tmp_pred_seg_array[:] = part

        tmp_pred_seg = nib.Nifti1Image(tmp_pred_seg_array, tmp_pred_seg.affine)
        print("------Resample Reverse------")
        pred_seg = self.dataloader.change_sacping(
            tmp_pred_seg,
            new_size=img_in.shape,
            order=0,
            force_affine=img_in.affine,
            dtype=np.uint8,
        )
        pred_seg = self.undo_canonical(pred_seg, img_in_ori)
        print("------Saving Result------")
        qform = pred_seg.get_qform()
        pred_seg.set_qform(qform)
        sform = pred_seg.get_sform()
        pred_seg.set_sform(sform)
        nib.save(pred_seg, self.nii_out_path)
        print("------Clean TMP Folder------")
        self.clean_tmp_folder()


if __name__ == "__main__":
    # nii_input_path = r"D:\data\Abdoman\dcm\060_230426360573\abdoman_5.nii.gz"
    model_folder = r"D:\workspace\TotalSegmentator-1_bugfixes\ckpt\nnunet\results\nnUNet\3d_fullres"
    # tmp_folder = r"D:\data\Abdoman\dcm\060_230426360573\tmp_debug"
    # nii_out_path = r"D:\data\Abdoman\dcm\060_230426360573\test_debug_5.nii.gz"
    nii_input_folder = r"D:\data\Abdoman\yiling_example\ori\image\export\125mm_nii"
    tmp_folder = r"D:\data\Abdoman\yiling_example\ori\image\export\tmp"
    # nii_output_folder = (
    #     r"D:\data\Abdoman\Task08_HepaticVessel\Task08_HepaticVessel\totalseg_labelsTr"
    # )
    nii_input_pathes = glob.glob(os.path.join(nii_input_folder, "*"))[:2]
    for nii_input_path in tqdm(nii_input_pathes):
        pid = nii_input_path.split(os.sep)[-1]
        print("Processing: ", pid)
        nii_out_path = os.path.join(
            r"D:\data\Abdoman\tmp",
            nii_input_path.split(os.sep)[-1],
        )
        if os.path.exists(nii_out_path):
            continue
        infer = Infer(
            nii_input_path=nii_input_path,
            model_folder=model_folder,
            tmp_folder=tmp_folder,
            nii_out_path=nii_out_path,
            if_fast=True,
            split_margin=20,
            all_in_gpu=True,
            mix_precision=True,
        )
        infer()
