#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from dataset.dataset import CtcSDF_dataset
from model import get_model
from reconstruction import reconstruct
from utils import export_pose_results


def resolve_global_index(data_root, sample_idx, data_sample=None):
    if data_sample is None:
        return int(sample_idx)

    sample_file = osp.join(data_root, f"test_sample_{data_sample}.json")
    if osp.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_map = json.load(f)
        all_indices = []
        for values in sample_map.values():
            all_indices.extend(values)
        if sample_idx < 0 or sample_idx >= len(all_indices):
            raise IndexError(
                f"sample_idx={sample_idx} out of range for data_sample={data_sample}; "
                f"valid range is [0, {len(all_indices)-1}]"
            )
        return int(all_indices[sample_idx])
    return int(sample_idx)


def load_model(model_path, device):
    model = get_model(is_train=False, batch=1, num_sample_points=2048, clamp_dist=0.05).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["network"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                stripped[key[len("module."):]] = value
            else:
                stripped[key] = value
        model.load_state_dict(stripped)
    model.eval()
    return model


def move_batch_to_device(inputs, metas, device):
    for key, value in inputs.items():
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], torch.Tensor):
                    value[i] = value[i].to(device, non_blocking=True)
        elif isinstance(value, torch.Tensor):
            inputs[key] = value.to(device, non_blocking=True)

    for key, value in metas.items():
        if key in ("id", "obj_id"):
            continue
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], torch.Tensor):
                    value[i] = value[i].to(device, non_blocking=True)
        elif isinstance(value, torch.Tensor):
            metas[key] = value.to(device, non_blocking=True)


def main():
    parser = argparse.ArgumentParser(description="Run network inference/reconstruction for a single test sample.")
    parser.add_argument("--sample-idx", type=int, required=True, help="Index inside full test split.")
    parser.add_argument(
        "--data-sample",
        type=int,
        default=None,
        help="Optional sampling config id. If omitted, sample-idx is interpreted as global test index.",
    )
    parser.add_argument(
        "--recon-mode",
        choices=["icp", "free"],
        default="icp",
        help="Reconstruction export mode.",
    )
    parser.add_argument(
        "--model-path",
        default=osp.join(osp.dirname(__file__), "ckpt", "snapshot_1599.pth.tar"),
        help="Checkpoint path.",
    )
    parser.add_argument("--device", default=None, help="cuda:0/cpu; default auto-detect.")
    args = parser.parse_args()

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    data_root = osp.join(repo_root, "dataset", "data")
    global_idx = resolve_global_index(data_root, args.sample_idx, data_sample=args.data_sample)

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    results_root = osp.join(osp.dirname(__file__), "results")
    obj_pose_result_dir = osp.join(results_root, "obj_pose_results")
    hand_pose_result_dir = osp.join(results_root, "hand_pose_results")
    os.makedirs(obj_pose_result_dir, exist_ok=True)
    os.makedirs(hand_pose_result_dir, exist_ok=True)

    dataset = CtcSDF_dataset(
        task="test",
        sdf_sample=2048,
        sdf_scale=6.2,
        clamp=0.05,
        input_image_size=(256, 256),
        start=global_idx,
        end=global_idx + 1,
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = load_model(args.model_path, device)

    with torch.no_grad():
        inputs, metas = next(iter(loader))
        print(inputs.keys())
        print(metas.keys())
        if device != "cpu":
            move_batch_to_device(inputs, metas, device)
        sdf_feat, hand_pose_results, obj_pose_results = model(inputs, targets=None, metas=metas, mode="test")
        export_pose_results(obj_pose_result_dir, obj_pose_results, metas)
        export_pose_results(hand_pose_result_dir, hand_pose_results, metas)
        reconstruct(
            metas["id"],
            model.obj_sdf_head,
            sdf_feat,
            metas,
            hand_pose_results,
            obj_pose_results,
            results_root=results_root,
            recon_mode=args.recon_mode,
        )

    print(
        f"[test_single] done sample_idx={args.sample_idx}, global_idx={global_idx}, "
        f"id={metas['id'][0]}, recon_mode={args.recon_mode}"
    )


if __name__ == "__main__":
    main()
