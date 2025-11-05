import os
import sys
import argparse
import socket
import signal
from tqdm import tqdm
from loguru import logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from utils import export_pose_results, reduce_tensor
from networks.training_utils import Trainer, Tester
import os.path as osp
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group, all_reduce


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # if not args.gpu_ids:
    #     assert 0, "Please set propoer gpu ids"

    # if '-' in args.gpu_ids:
    #     gpus = args.gpu_ids.split('-')
    #     gpus[0] = int(gpus[0])
    #     gpus[1] = int(gpus[1]) + 1
    #     args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12330"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   
def main(rank):
    cudnn.benchmark = True
    local_rank = rank
    ddp_setup(local_rank, 2)
    log_dir = osp.join(osp.dirname(__file__), 'logs')
    world_size = torch.distributed.get_world_size()    

    logger.info('Distributed Process %d, Total %d.' % (local_rank, world_size))
    if local_rank == 0:
        writer_dict = {'writer': SummaryWriter(log_dir = log_dir), 'train_global_steps': 0}

    trainer = Trainer(batch_size=256, num_gpus=world_size, exp='hmano_osdf')
    trainer._make_batch_generator()
    trainer._make_model(local_rank)
    end_epoch = 2800
    for epoch in range(trainer.start_epoch, end_epoch):
        trainer.total_timer.tic()
        trainer.read_timer.tic()
        trainer.train_sampler.set_epoch(epoch)
        
        for itr, (inputs, targets, metas) in enumerate(trainer.batch_generator):
            trainer.set_lr(epoch, itr)
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda(non_blocking=True)            
                else:
                    inputs[k] = inputs[k].cuda(non_blocking=True)
            for k, v in targets.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        targets[k][i] = targets[k][i].cuda(non_blocking=True)
                else:
                    targets[k] = targets[k].cuda(non_blocking=True)
            metas['epoch'] = epoch
            for k, v in metas.items():
                if k != 'id' and k != 'epoch' and k != 'obj_id':
                    if isinstance(v, list):
                        for i in range(len(v)):
                            metas[k][i] = metas[k][i].cuda(non_blocking=True)
                    else:
                        metas[k] = metas[k].cuda(non_blocking=True)
            
            trainer.optimizer.zero_grad()
            loss, sdf_results, hand_pose_results, obj_pose_results = trainer.model(inputs, targets, metas, 'train')
            all_loss = sum(loss[k] for k in loss)
            torch.distributed.barrier()

            all_loss.backward()

            trainer.optimizer.step()
            torch.cuda.synchronize()            
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (trainer.total_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fs/epoch' % (trainer.total_timer.average_time * trainer.itr_per_epoch),
                ]

            record_dict = {}
            for k, v in loss.items():
                record_dict[k] = reduce_tensor(v.detach(), world_size) * 1000.
            screen += ['%s: %.3f' % ('loss_' + k, v) for k, v in record_dict.items()]

            if local_rank == 0:
                tb_writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                if itr % 10 == 0:
                    trainer.logger.info(' '.join(screen))
                    for k, v in record_dict.items():
                        tb_writer.add_scalar('loss_' + k, v, global_steps)
                    tb_writer.add_scalar('lr', trainer.get_lr(), global_steps)
                    writer_dict['train_global_steps'] = global_steps + 10

            trainer.total_timer.toc()
            trainer.total_timer.tic()
            trainer.read_timer.tic()
            
        if local_rank == 0 and (epoch % 20 == 0 or epoch == end_epoch - 1):
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)
            writer_dict['writer'].close()
            
            
if __name__ == '__main__':
    mp.spawn(main, args=(), nprocs=2)