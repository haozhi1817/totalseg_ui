"""
Author: HaoZhi
Date: 2024-07-08 15:26:58
LastEditors: HaoZhi
LastEditTime: 2024-07-08 15:28:53
Description: 
"""
import os
import sys

if "win" in sys.platform:
    # fix for windows platform
    import pathos

    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

from multiprocessing import Pool

import torch
import numpy as np

from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.inference.segmentation_export import save_segmentation_nifti


class Predictor(object):
    def __init__(self) -> None:
        pass

    def preprocess_save_to_queue(
        self,
        preprocess_fn,
        q,
        list_of_lists,
        output_files,
    ):
        errors_in = []
        for i, l in enumerate(list_of_lists):
            print("debug:  ", i, l, output_files[i])
            try:
                output_file = output_files[i]
                # print("preprocessing", output_file)
                d, _, dct = preprocess_fn(l)
                # print(d.shape)
                if np.prod(d.shape) > (
                    2e9 / 4 * 0.85
                ):  # *0.85 just to be save, 4 because float32 is 4 bytes
                    print(
                        "This output is too large for python process-process communication. "
                        "Saving output temporarily to disk"
                    )
                    np.save(output_file[:-7] + ".npy", d)
                    d = output_file[:-7] + ".npy"
                q.put((output_file, (d, dct)))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print("error in", l)
                print(e)
        q.put("end")
        if len(errors_in) > 0:
            print("There were some errors in the following cases:", errors_in)
            print("These cases were ignored.")
        else:
            print("This worker has ended successfully, no errors to report")

    def preprocess_multithreaded(
        self,
        trainer,
        list_of_lists,
        output_files,
        num_processes=2,
        segs_from_prev_stage=None,
    ):
        if segs_from_prev_stage is None:
            segs_from_prev_stage = [None] * len(list_of_lists)

        num_processes = min(len(list_of_lists), num_processes)

        assert isinstance(trainer, nnUNetTrainer)
        q = Queue(1)
        processes = []
        for i in range(num_processes):
            pr = Process(
                target=self.preprocess_save_to_queue,
                args=(
                    trainer.preprocess_patient,
                    q,
                    list_of_lists[i::num_processes],
                    output_files[i::num_processes],
                ),
            )
            pr.start()
            processes.append(pr)

        try:
            end_ctr = 0
            while end_ctr != num_processes:
                item = q.get()
                if item == "end":
                    end_ctr += 1
                    continue
                else:
                    yield item

        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()  # this should not happen but better safe than sorry right
                p.join()

            q.close()

    def predict(
        self,
        trainer,
        params,
        inputs_list,
        output_list,
        num_threads_preprocessing,
        num_threads_nifti_save,
        all_in_gpu,
        mixed_precision,
    ):
        pool = Pool(num_threads_nifti_save)
        result = []
        torch.cuda.empty_cache()

        preprocessing = self.preprocess_multithreaded(
            trainer,
            inputs_list,
            output_list,
            num_threads_preprocessing,
            segs_from_prev_stage=None,
        )

        for preprocessed in preprocessing:
            output_filename, (d, dct) = preprocessed
            # print("debug: ", output_filename, d, dct)
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data
            all_softmax_outputs = np.zeros(
                (len(params), trainer.num_classes, *d.shape[1:]), dtype=np.float16
            )
            all_seg_outputs = np.zeros((len(params), *d.shape[1:]), dtype=int)
            for i, p in enumerate(params):
                trainer.load_checkpoint_ram(p, False)
                res = trainer.predict_preprocessed_data_return_seg_and_softmax(
                    d,
                    do_mirroring=False,
                    mirror_axes=trainer.data_aug_params["mirror_axes"],
                    use_sliding_window=True,
                    step_size=0.5,
                    use_gaussian=True,
                    all_in_gpu=all_in_gpu,
                    mixed_precision=mixed_precision,
                )
                if len(params) > 1:
                    all_softmax_outputs[i] = res[1]
                all_seg_outputs[i] = res[0]
                seg = all_seg_outputs[0]
                # print("applying transpose_backward")
                transpose_forward = trainer.plans.get("transpose_forward")
                if transpose_forward is not None:
                    transpose_backward = trainer.plans.get("transpose_backward")
                    seg = seg.transpose([i for i in transpose_backward])
                # print("debug: ", seg.shape)
                # print("initializing segmentation export")
                result.append(
                    pool.starmap_async(
                        save_segmentation_nifti, ((seg, output_filename, dct, 0, None),)
                    )
                )
        _ = [i.get() for i in result]
        pool.close()
        pool.join()
