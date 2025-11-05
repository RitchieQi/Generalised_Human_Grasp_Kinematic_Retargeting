import os
import os.path as osp
import math
import glob
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.optim
from networks.model import get_model
from utils import Timer
from dataset import CtcSDF_dataset
class Base_trainer(ABC):
    def __init__(self, exp, log_name = 'logs.txt'):
        self.current_epoch = 0
        self.total_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        self.exp = exp
        self.logger = logger
        self.logger.add(osp.join(exp,'logs', log_name))
        self.model_dir = osp.join(osp.dirname(__file__), self.exp, 'models')

    
    @abstractmethod
    def _make_batch_generator(self):
        return

    @abstractmethod
    def _make_model(self):
        return
    
class Trainer(Base_trainer):
    def __init__(self, batch_size, num_gpus, exp,  lr=1e-4):
        super(Trainer, self).__init__(exp=exp)
        self.lr = lr
        self.lr_dec_epoch = [600,1200]
        self.lr_dec_factor = 0.5
        self.train_batch_size = batch_size
        self.num_gpus = num_gpus
    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        return optimizer        
    def save_model(self, state, epoch):
        file_path = osp.join(self.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info('Model saved at {}'.format(file_path))
    def load_model(self, model, optimizer, checkpoint_file=None):
        model_file_list = glob.glob(osp.join(self.model_dir, 'snapshot_*.pth.tar'))
        if len(model_file_list) == 0:
            if checkpoint_file is not None and osp.exists(checkpoint_file):
                ckpt = torch.load(checkpoint_file, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['network'])
                start_epoch = 0
                self.logger.info('Load checkpoint from {}'.format(checkpoint_file))
                return start_epoch, model, optimizer
            else:
                start_epoch = 0
                self.logger.info('start training from scratch')
                return start_epoch, model, optimizer
        else:
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
            ckpt_path = osp.join(self.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['network'])
            optimizer.load_state_dict(ckpt['optimizer'])
            self.logger.info('Continue training from epoch {}'.format(start_epoch))
            return start_epoch, model, optimizer
    def set_lr(self, epoch, iter_num):
        cur_lr = self.lr
        for i in range(len(self.lr_dec_epoch)):
            if epoch >= self.lr_dec_epoch[i]:
                cur_lr = cur_lr * self.lr_dec_factor
        
        self.optimizer.param_groups[0]['lr'] = cur_lr
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    def _make_batch_generator(self):
        self.logger.info('Creating dataset...')
        self.trainset_loader = CtcSDF_dataset(task='train', 
                                              sdf_sample=2048, 
                                              sdf_scale=6.2, 
                                              clamp=0.05, 
                                              input_image_size=(256, 256))
        self.itr_per_epoch = math.ceil(len(self.trainset_loader) / self.num_gpus / self.train_batch_size)
        self.train_sampler = DistributedSampler(self.trainset_loader)
        self.batch_generator = DataLoader(self.trainset_loader, 
                                            batch_size=self.train_batch_size, 
                                            shuffle=False,
                                            num_workers=6,
                                            pin_memory=True,
                                            sampler=self.train_sampler,
                                            drop_last=True,
                                            persistent_workers=False)
    def _make_model(self, local_rank):
        self.logger.info("Creating graph and optimizer...")
        model = get_model(is_train=True, batch=self.train_batch_size, num_sample_points=2048, clamp_dist=0.05)
        model = model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        optimizer = self.get_optimizer(model)
        model = NativeDDP(model, device_ids=[local_rank], output_device=local_rank)
        model.train()
        start_epoch, model, optimizaer = self.load_model(model, optimizer)
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        
class Tester(Base_trainer):
    def __init__(self, local_rank, test_epoch, exp):
        self.local_rank = local_rank
        self.test_epoch = test_epoch
        self.num_testset_samples = 5928
        self.num_gpus = 2
        self.test_batch_size = 1

        super(Tester, self).__init__(log_name='test_logs.txt',exp=exp)
    
    def _make_batch_generator(self):
        start_points = []
        end_points = []
        
        total_test_samples = self.num_testset_samples
        division = total_test_samples // self.num_gpus
        for i in range(self.num_gpus):
            start_point = i * division
            if i != self.num_gpus - 1:
                end_point = (i + 1) * division
            else:
                end_point = total_test_samples
            start_points.append(start_point)
            end_points.append(end_point)
        self.logger.info(f"Creating dataset from {start_points[self.local_rank]} to {end_points[self.local_rank]}")
        self.testset_loader = CtcSDF_dataset(task='test', 
                                             sdf_sample=2048, 
                                             sdf_scale=6.2, 
                                             clamp=0.05, 
                                             input_image_size=(256, 256),
                                             start=start_points[self.local_rank],
                                             end=end_points[self.local_rank])
        self.itr_per_epoch = math.ceil(len(self.testset_loader) / self.num_gpus / self.test_batch_size)
        self.batch_generator = DataLoader(dataset=self.testset_loader, batch_size=self.test_batch_size, shuffle=False, num_workers=6, pin_memory=True, drop_last=False, persistent_workers=False)
    
    def _make_model(self, local_rank):
        model_path = osp.join(self.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert osp.exists(model_path), 'Model snapshot not found at {}'.format(model_path)
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        self.logger.info('Creating computation graph...')
        model = get_model(is_train=False, batch=self.test_batch_size, num_sample_points=2048, clamp_dist=0.05)
        model = model.cuda()
        model = NativeDDP(model, device_ids=[local_rank], output_device=local_rank)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        self.model = model

        