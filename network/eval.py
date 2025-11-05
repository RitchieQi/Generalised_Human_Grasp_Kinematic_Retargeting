import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
from multiprocessing import Process, Queue
import pandas as pd
import trimesh
import shutil
import os.path as osp
from dataset.dataset import CtcSDF_dataset
from tqdm import tqdm

def evaluate(queue, db, output_dir):
    for idx, _ in tqdm(enumerate(range(db.__len__()))):
        error_dict = db._evaluate(output_dir, idx)
        queue.put([tuple(error_dict.values())])

def main():
    # test_set = CtcSDF_dataset(task='test', 
    #                           sdf_sample=2048,
    #                           sdf_scale=6.2,
    #                           clamp=0.05,
    #                           input_image_size=(256, 256))
    data_root = osp.join(osp.dirname(__file__), '..', 'CtcSDF', 'data')
    summaries = []
    output_dir = osp.join(osp.dirname(__file__), 'hmano_osdf')

    num_proc = 10
    num_testset_samples = 10# 5928

    start_points = []
    end_points = []
    division = num_testset_samples // num_proc
    for i in range(num_proc):
        start_point = i * division
        if i != num_proc - 1:
            end_point = start_point + division
        else:
            end_point = num_testset_samples
        start_points.append(start_point)
        end_points.append(end_point)
    
    queue = Queue()
    process_list = []
    for i in range(num_proc):
        test_set = eval('CtcSDF_dataset')(start_points[i], end_points[i], task='test', 
                                          sdf_sample=2048, 
                                          sdf_scale=6.2, 
                                          clamp=0.05, 
                                          input_image_size=(256, 256))
        p = Process(target=evaluate, args=(queue, test_set, output_dir))
        p.start()
        process_list.append(p)

    summaries = []
    for p in process_list:
        while p.is_alive():
            while False == queue.empty():
                data = queue.get()
                summaries.append(data[0])

    for p in process_list:
        p.join()

    
    # summaries = sorted(summaries, key=lambda result: result[0])
    summary_filename = 'eval_results.txt'

    with open(osp.join(output_dir, summary_filename), 'w') as f:
        eval_results = [[] for i in range(8)]
        name_list = ['sampel_id', 
                     'chamfer_obj', 
                     'fscore_obj_5', 
                     'fscore_obj_10', 
                     'mano_joint', 
                     'obj_center', 
                     'chamfer_hand', 
                     'fscore_hand_5', 
                     'fscore_hand_10']
        data_list = []
        for i, result in enumerate(summaries):
            data_sample = [result[0]]
            for i in range(8):
                if result[i+1] is not None:
                    eval_results[i].append(result[i + 1])
                    data_sample.append(result[i + 1].round(3))
                else:
                    data_sample.append(result[i + 1])
            data_list.append(data_sample)
        f.write(pd.DataFrame(data_list, columns=name_list, index=[''] * len(summaries), dtype=str).to_string())
        f.write('\n')

        for idx, _ in enumerate(eval_results):
            new_array = []
            for number in eval_results[idx]:
                if not np.isnan(number):
                    new_array.append(number)
            eval_results[idx] = new_array
        
        mean_chamfer_obj = "mean obj chamfer: {}\n".format(np.mean(eval_results[0]))
        median_chamfer_obj = "median obj chamfer: {}\n".format(np.median(eval_results[0]))
        fscore_obj_1 = "f-score obj @ 1mm: {}\n".format(np.mean(eval_results[1]))
        fscore_obj_5 = "f-score obj @ 5mm: {}\n".format(np.mean(eval_results[2]))          
        mean_mano_joint_err = "mean mano joint error: {}\n".format(np.mean(eval_results[3]))
        mean_obj_center_err = "mean obj center error: {}\n".format(np.mean(eval_results[4]))
        mean_chamfer_hand = "mean hand chamfer: {}\n".format(np.mean(eval_results[5]))
        median_chamfer_hand = "median hand chamfer: {}\n".format(np.median(eval_results[5]))
        fscore_hand_1 = "f-score hand @ 1mm: {}\n".format(np.mean(eval_results[6]))
        fscore_hand_5 = "f-score hand @ 5mm: {}\n".format(np.mean(eval_results[7]))

        print(mean_chamfer_hand); f.write(mean_chamfer_hand)
        print(median_chamfer_hand); f.write(median_chamfer_hand)
        print(fscore_hand_1); f.write(fscore_hand_1)
        print(fscore_hand_5); f.write(fscore_hand_5)
        print(mean_chamfer_obj); f.write(mean_chamfer_obj)
        print(median_chamfer_obj); f.write(median_chamfer_obj)
        print(fscore_obj_1); f.write(fscore_obj_1)
        print(fscore_obj_5); f.write(fscore_obj_5)
        print(mean_mano_joint_err); f.write(mean_mano_joint_err)
        print(mean_obj_center_err); f.write(mean_obj_center_err)

if __name__ == '__main__':
    main()