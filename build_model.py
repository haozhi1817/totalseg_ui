"""
Author: HaoZhi
Date: 2024-07-10 14:56:21
LastEditors: HaoZhi
LastEditTime: 2024-07-10 14:57:04
Description: 
"""
"""
Author: HaoZhi
Date: 2024-07-08 14:19:53
LastEditors: HaoZhi
LastEditTime: 2024-07-09 10:22:54
Description: 
"""
import os
import pickle
import pkgutil
import importlib

import torch

import nnunet


class BuildModel(object):
    def __init__(self) -> None:
        pass

    def load_pkl(self, pkl_path):
        with open(pkl_path, "rb") as f:
            info = pickle.load(f)
        return info

    def recursive_find_python_class(self, folder, trainer_name, current_module):
        tr = None
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if not ispkg:
                m = importlib.import_module(current_module + "." + modname)
                if hasattr(m, trainer_name):
                    tr = getattr(m, trainer_name)
                    break
        if tr is None:
            for importer, modname, ispkg in pkgutil.iter_modules(folder):
                if ispkg:
                    next_current_module = current_module + "." + modname
                    tr = self.recursive_find_python_class(
                        [os.path.join(folder[0], modname)],
                        trainer_name,
                        next_current_module,
                    )
                    if tr is not None:
                        break

        return tr

    def __call__(self, model_folder, fp16):
        model_pkl_path = os.path.join(
            model_folder, "fold_0", "model_final_checkpoint.model.pkl"
        )
        param_path = os.path.join(
            model_folder, "fold_0", "model_final_checkpoint.model"
        )

        pkl_info = self.load_pkl(model_pkl_path)
        model_init = pkl_info["init"]
        model_name = pkl_info["name"]
        model_plan = pkl_info["plans"]
        search_in = os.path.join(nnunet.__path__[0], "training", "network_training")
        tr = self.recursive_find_python_class(
            [search_in], model_name, current_module="nnunet.training.network_training"
        )
        trainer = tr(*model_init)
        if fp16 is not None:
            trainer.fp16 = fp16
        trainer.process_plans(model_plan)
        trainer.output_folder = model_folder
        trainer.output_folder_base = model_folder
        trainer.update_fold(0)
        trainer.initialize(False)

        params = torch.load(param_path, map_location="cpu")
        return trainer, [params]


if __name__ == "__main__":
    model_folder = r"C:\Users\hxk\.totalsegmentator\nnunet\results\nnUNet\3d_fullres\Task251_TotalSegmentator_part1_organs_1139subj\nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1"
    builder = BuildModel()
    model, params = builder(model_folder, fp16=True)
    print(model)
    print(params)
