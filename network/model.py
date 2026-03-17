from network.networks import ResNetBackbone, UNet, SDFHead, ManoHead, kinematic_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.utils import soft_argmax, decode_volume, get_mano_preds


class model(nn.Module):
    def __init__(self, backbone, neck, obj_sdf_head, mano_head, volume_head, train_batch_size, num_sample_points, clamp_dist, test_batch_size):
        super(model, self).__init__()
        self.backbone = backbone
        self.dim_backbone_feat = 512
        self.neck = neck
        self.obj_sdf_head = obj_sdf_head
        self.mano_head = mano_head
        self.sdf_latent = 256
        self.train_batch_size = train_batch_size
        self.num_sample_points = num_sample_points
        self.clamp_dist = clamp_dist
        self.volume_head = volume_head
        self.image_size = (256, 256)
        self.heatmap_size = (64, 64, 64)
        self.depth_dim = 0.28
        self.obj_point_latent = 6
        self.recon_scale = 6.2
        self.pose_epoch = 0
        self.test_batch_size = test_batch_size
        self.backbone_2_sdf = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.sdf_encoder = nn.Linear(512, self.sdf_latent)        
        
        if self.mano_head is not None:
            self.backbone_2_mano = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
            
        self.loss_l1 = nn.L1Loss(reduction='sum')
        self.loss_l2 = nn.MSELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        
    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_img = inputs['img']
            sdf_data = targets['obj_sdf']
            # cls_data = targets['obj_label']
            mask_obj = torch.ones(self.train_batch_size * self.num_sample_points).unsqueeze(1).to(input_img.device)

            sdf_data = sdf_data.reshape(self.train_batch_size * self.num_sample_points, -1)
            # cls_data = cls_data.to(torch.long).reshape(self.train_batch_size * self.num_sample_points)
            xyz_points = sdf_data[:, 0:-2]
            sdf_gt_obj = sdf_data[:, -1].unsqueeze(1)
            sdf_gt_obj = torch.clamp(sdf_gt_obj, -self.clamp_dist, self.clamp_dist)

            backbone_feat = self.backbone(input_img)
            
            volume_results = {}
            hm_feat = self.neck(backbone_feat)
            hm_pred = self.volume_head(hm_feat)
            hm_pred, hm_conf = soft_argmax(self.heatmap_size, hm_pred, num_joints=1)
            volume_joint_preds = decode_volume(self.image_size, self.heatmap_size,self.depth_dim, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
            volume_results['joints'] = volume_joint_preds
            
            mano_feat = self.backbone_2_mano(backbone_feat)
            mano_feat = mano_feat.mean(3).mean(2)
            hand_pose_results = self.mano_head(mano_feat)
            hand_pose_results = get_mano_preds(hand_pose_results, self.image_size, metas['cam_intr'], metas['hand_center_3d'])

            obj_pose_results = {}
            obj_transform = torch.zeros(self.train_batch_size, 4, 4).to(input_img.device)
            obj_transform[:, :3, 3] = volume_joint_preds[:, 0, :] - metas['hand_center_3d']
            obj_transform[:, 3, 3] = 1
            obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
            
            obj_pose_results['global_trans'] = obj_transform
            obj_pose_results['center'] = volume_joint_preds
            obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
            
            sdf_feat = self.backbone_2_sdf(backbone_feat)
            sdf_feat = sdf_feat.mean(3).mean(2)
            sdf_feat = self.sdf_encoder(sdf_feat)
            sdf_feat = sdf_feat.repeat_interleave(self.num_sample_points, dim=0)
            
            obj_points = kinematic_embedding(self.obj_point_latent, self.recon_scale, xyz_points, self.num_sample_points, obj_pose_results, 'obj')
            obj_points = obj_points.reshape(-1, self.obj_point_latent)
            obj_sdf_decoder_inputs = torch.cat([sdf_feat, obj_points], dim=1)

            sdf_obj, _ = self.obj_sdf_head(obj_sdf_decoder_inputs)
            sdf_obj = torch.clamp(sdf_obj, min=-self.clamp_dist, max=self.clamp_dist)

            sdf_results = {}
            sdf_results['obj'] = sdf_obj
            
            loss = {}
            loss['obj_sdf'] = 0.5 * self.loss_l1(sdf_obj * mask_obj, sdf_gt_obj * mask_obj) / mask_obj.sum()
            
            valid_idx = hand_pose_results['vis']
            loss['mano_joints'] = 0.5 * self.loss_l2(valid_idx.unsqueeze(-1) * hand_pose_results['joints'], valid_idx.unsqueeze(-1) * targets['hand_joints_3d'])
            loss['mano_shape'] = 5e-7 * self.loss_l2(hand_pose_results['shape'], torch.zeros_like(hand_pose_results['shape']))
            loss['mano_pose'] = 5e-5 * self.loss_l2(valid_idx * (hand_pose_results['pose'][:, 3:] - hand_pose_results['mean_pose']), valid_idx * torch.zeros_like(hand_pose_results['pose'][:, 3:]))
         
            volume_joint_targets = targets['obj_center_3d'].unsqueeze(1)
            loss['volume_joint'] = 0.5 * self.loss_l2(volume_joint_preds, volume_joint_targets)
            return loss, sdf_results, hand_pose_results, obj_pose_results
        else:
            with torch.no_grad():
                input_img = inputs['img']
                backbone_feat = self.backbone(input_img)
                
                sdf_feat = self.backbone_2_sdf(backbone_feat)
                sdf_feat = sdf_feat.mean(3).mean(2)
                sdf_feat = self.sdf_encoder(sdf_feat)
                
                volume_results = {}
                hm_feat = self.neck(backbone_feat)
                hm_pred = self.volume_head(hm_feat)
                hm_pred, hm_conf = soft_argmax(self.heatmap_size, hm_pred, num_joints=1)
                volume_joint_preds = decode_volume(self.image_size, self.heatmap_size,self.depth_dim, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
                volume_results['joints'] = volume_joint_preds
                
                mano_feat = self.backbone_2_mano(backbone_feat)
                mano_feat = mano_feat.mean(3).mean(2)
                hand_pose_results = self.mano_head(mano_feat)
                hand_pose_results = get_mano_preds(hand_pose_results, self.image_size, metas['cam_intr'], metas['hand_center_3d'])
                
                obj_pose_results = {}
                obj_transform = torch.zeros(self.test_batch_size, 4, 4).to(input_img.device)
                obj_transform[:, :3, 3] = volume_joint_preds[:, 0, :] - metas['hand_center_3d']
                obj_transform[:, 3, 3] = 1                
                obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                # obj_corners = metas['obj_rest_corners_3d'] + volume_joint_preds

                obj_pose_results['global_trans'] = obj_transform
                obj_pose_results['center'] = volume_joint_preds
                obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]

            return sdf_feat, hand_pose_results, obj_pose_results
        
        
def get_model(is_train, batch, num_sample_points, clamp_dist):
    num_resnet_layers = 18
    backbone = ResNetBackbone(num_resnet_layers)
    if is_train:
        backbone.init_weights()
        train_batch_size = batch
        test_batch_size = None
    else:
        test_batch_size = batch
        train_batch_size = None
    neck_inplanes = 512
    neck = UNet(neck_inplanes, 256, 3)
    
    sdf_latent = 256
    obj_point_latent = 6
    layers = 5
    dims = [512 for i in range(layers - 1)]
    dropout = [i for i in range(layers - 1)]
    norm_layers = [i for i in range(layers - 1)]
    dropout_prob = 0.2
    latent_in = [(layers - 1) // 2]
    obj_sdf_head = SDFHead(sdf_latent, obj_point_latent, dims, dropout, dropout_prob, norm_layers, latent_in, False)
    mano_head = ManoHead(depth=False)
    volume_head = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
    
    model_ = model(backbone, neck, obj_sdf_head, mano_head, volume_head, train_batch_size, num_sample_points, clamp_dist, test_batch_size)

    return model_

if __name__ == '__main__':
    from dataset.dataset import CtcSDF_dataset
    from torch.utils.data import DataLoader
    def _to(x, dev):
        if isinstance(x, torch.Tensor): return x.to(dev)
        if isinstance(x, list): return [_to(y, dev) for y in x]
        if isinstance(x, dict): return {k: _to(v, dev) for k, v in x.items()}
        return x

    dataset = CtcSDF_dataset(task='train', sdf_sample=2048, sdf_scale=6.2, clamp=0.05, input_image_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model_ = get_model(is_train=True, batch=4, num_sample_points=2048, clamp_dist=0.05)
    model_.cuda()
    total_params = sum(p.numel() for p in model_.parameters())
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")
    inputs, targets, metas = next(iter(dataloader))
    inputs = _to(inputs, 'cuda')
    targets = _to(targets, 'cuda')
    metas = _to(metas, 'cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss, sdf_results, hand_pose_results, obj_pose_results = model_(inputs, targets, metas, 'train')
    end.record()
    torch.cuda.synchronize()
    print(f"Forward pass time: {start.elapsed_time(end):.2f} ms")

    # print(loss)
    # print(sdf_results.keys())
    # print(hand_pose_results.keys())
    # print(obj_pose_results.keys())
    
    