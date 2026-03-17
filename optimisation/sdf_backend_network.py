import os
import os.path as osp

import torch

from dataset.dataset import CtcSDF_dataset
from network.sdf import DeepSDF_object


class NetworkSDFQueryBridge:
    """
    Bridge optimisation sample ids to network object-branch SDF queries.
    """

    def __init__(self, model_path=None, device="cpu", use_pred_center_offset=False):
        if model_path is None:
            repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
            model_path = osp.join(repo_root, "network", "ckpt", "snapshot_1599.pth.tar")
        if not osp.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        self.device = device
        self.use_pred_center_offset = bool(use_pred_center_offset)
        self.model = DeepSDF_object(model_path=model_path, device=device)
        self.dataset = CtcSDF_dataset(task="test")
        self.id_to_index = {
            image_item["file_name"]: i for i, image_item in enumerate(self.dataset.json_data["images"])
        }
        self.current_id = None

    def _to_device_metas(self, metas):
        metas_out = {}
        for key, value in metas.items():
            if isinstance(value, torch.Tensor):
                metas_out[key] = value.unsqueeze(0).to(self.device)
            else:
                metas_out[key] = value
        return metas_out

    def set_sample(self, sample_id):
        if sample_id == self.current_id:
            return
        if sample_id not in self.id_to_index:
            raise KeyError(f"Sample id not found in CtcSDF test set: {sample_id}")

        sample_index = self.id_to_index[sample_id]
        inputs, metas = self.dataset[sample_index]
        input_img = inputs["img"].unsqueeze(0).to(self.device)
        metas = self._to_device_metas(metas)
        self.model.set_context(input_img=input_img, metas=metas)
        self.current_id = sample_id

    def set_sample_from_input_pack(self, input_pack):
        sample_id = input_pack["file_name"]
        self.set_sample(sample_id)
        return self.get_current_obj_translation(device=self.device, dtype=torch.float32)

    def _get_current_obj_translation(self, device=None, dtype=None):
        pose = getattr(self.model, "_cached_obj_pose_results", None)
        if not isinstance(pose, dict):
            return None
        global_trans = pose.get("global_trans", None)
        if not isinstance(global_trans, torch.Tensor):
            return None
        if global_trans.dim() == 3:
            trans = global_trans[0, :3, 3]
        else:
            trans = global_trans[:3, 3]
        if device is None:
            device = trans.device
        if dtype is None:
            dtype = trans.dtype
        return trans.to(device=device, dtype=dtype)

    def get_current_obj_translation(self, device=None, dtype=None):
        return self._get_current_obj_translation(device=device, dtype=dtype)

    def query(self, points):
        # Keep optimization and SDF query in the same frame by default.
        # Predicted-center offset can be enabled for ablation.
        trans = (
            self._get_current_obj_translation(device=points.device, dtype=points.dtype)
            if self.use_pred_center_offset
            else None
        )
        query_points = points if trans is None else points + trans.view(*([1] * (points.dim() - 1)), 3)
        return self.model.sdf_and_gradient_with_context(
            query_points, create_graph=points.requires_grad
        )
