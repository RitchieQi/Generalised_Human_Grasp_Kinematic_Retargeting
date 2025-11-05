import os
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import torch
from loguru import logger
from optimisation_tmp.utils import export_pose_results
from training_utils import Tester
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_reduce

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args
def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12330"
   torch.cuda.set_device(rank) #rank
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   
def main(rank):
    local_rank = rank
    ddp_setup(local_rank, 1)
    device = 'cuda:%d' % local_rank
    torch.cuda.set_device(local_rank)
    exp = 'hmano_osdf'
    obj_pose_result_dir = osp.join(osp.dirname(__file__), exp, 'obj_pose_results')
    hand_pose_result_dir = osp.join(osp.dirname(__file__), exp, 'hand_pose_results')
    if not osp.exists(obj_pose_result_dir):
        os.makedirs(obj_pose_result_dir)
    if not osp.exists(hand_pose_result_dir):
        os.makedirs(hand_pose_result_dir)
    
    tester = Tester(local_rank=local_rank, exp=exp, test_epoch=1600-1)
    tester._make_batch_generator()
    tester._make_model(local_rank=local_rank)

    with torch.no_grad():
        for itr, (inputs, metas) in tqdm(enumerate(tester.batch_generator)):
            if itr > 10:
                break
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda(non_blocking=True)
                else:
                    inputs[k] = inputs[k].cuda(non_blocking=True)

            for k, v in metas.items():
                if k != 'id' and k != 'obj_id':
                    if isinstance(v, list):
                        for i in range(len(v)):
                            metas[k][i] = metas[k][i].cuda(non_blocking=True)
                    else:
                        metas[k] = metas[k].cuda(non_blocking=True)

            sdf_feat, hand_pose_results, obj_pose_results = tester.model(inputs, targets=None, metas=metas, mode='test')
            export_pose_results(obj_pose_result_dir, obj_pose_results, metas)
            export_pose_results(hand_pose_result_dir, hand_pose_results, metas)
            from reconstruction import reconstruct
            reconstruct(metas['id'], tester.model.module.obj_sdf_head, sdf_feat, metas, hand_pose_results, obj_pose_results)


if __name__ == '__main__':
    mp.spawn(main, args=(), nprocs=1)