import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
import json
from scipy.spatial.transform import Rotation as R
import torch
import time
import trimesh as tm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import cv2
from manopth.manolayer import ManoLayer
from tqdm import tqdm
import traceback
from multiprocessing import Process, Queue
from itertools import product
import os
from environment import Environment

def simulation_worker(robot,mu,data_sample, exp, q):
    try:
        env = Environment(robot=robot, mu=mu, exp= exp, repeat=1, render="DIRECT", data_sample=data_sample, task = 'test', test_sdf=False)
        env.run(record=False)
        q.put((robot, mu, exp, 'Done'))
    except Exception as e:
        q.put((robot, mu, exp, f"Error: {str(e)}"))
        
def run_simulation_mp():
    robot = ["Shadow","Barrett", "Robotiq","Allegro"]
    mus = [0.9]
    data_sample = 5
    cores = 10
    exp = ["genhand"] #genhand
    processes = []
    q = Queue()
    task_args = list(product(robot, mus, exp))
    for robot, mu, exp in task_args:
        p = Process(target=simulation_worker, args=(robot, mu, data_sample, exp, q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    while not q.empty():
        robot, mu, exp, status = q.get()
        print(f"Robot: {robot}, Mu: {mu}, Exp: {exp}, Status: {status}")

if __name__ == "__main__":
    run_simulation_mp()