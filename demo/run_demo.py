#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_pipeline(sdf_source, method, cluster_method):
    cluster_map = {
        "kmeans": "kmeans",
        "hdbscan": "hdbscan",
    }

    if method == "genhand":
        if cluster_method is None:
            raise ValueError("--cluster-method is required when --method genhand.")
        opt_task = "GH"
        sim_exp = "genhand"
        contact_target_method = cluster_map[cluster_method]
    else:
        opt_task = "NV"
        sim_exp = "nv"
        contact_target_method = "hdbscan"

    if sdf_source == "gt_pv":
        return {
            "sdf_backend": "mesh",
            "source": "ycb",
            "precept": False,
            "run_network_stage": False,
            "opt_task": opt_task,
            "sim_exp": sim_exp,
            "contact_target_method": contact_target_method,
        }
    if sdf_source == "recon_pv":
        return {
            "sdf_backend": "mesh",
            "source": "sdf",
            "precept": True,
            "run_network_stage": True,
            "recon_mode": "icp",
            "pred_mesh_mode": "icp",
            "opt_task": opt_task,
            "sim_exp": sim_exp,
            "contact_target_method": contact_target_method,
        }
    if sdf_source == "model_direct":
        raise ValueError(
            "sdf_source=model_direct is temporarily disabled for this release. "
            "Use gt_pv or recon_pv."
        )
    raise ValueError(f"Unknown sdf_source: {sdf_source}")


def build_opt_code(
    robot_name,
    sample_idx,
    device,
    opt_render,
    mu,
    pipeline,
    contact_threshold,
):
    use_plotly = opt_render in ("plotly", "pyrender")
    return f"""
from optimisation.Optimize import Optimization
from optimisation.dataset_opt import dexycb_testfullfeed
from optimisation.robot import Shadow, Allegro, Barrett, Robotiq
import torch

robot_cls_map = {{
    "Shadow": Shadow,
    "Allegro": Allegro,
    "Barrett": Barrett,
    "Robotiq": Robotiq,
}}
robot = robot_cls_map[{robot_name!r}](batch=1, device={device!r})
sdf_prepare_fn = None
sdf_query_fn = None

dataset = dexycb_testfullfeed(
    load_mesh=True,
    pc_sample=1024,
    data_sample=None,
    precept={pipeline["precept"]},
    pred_mesh_mode={pipeline.get("pred_mesh_mode", "icp")!r},
)
opt = Optimization(
    robot=robot,
    dataset=dataset,
    device={device!r},
    maximum_iter=[2000, 1, 1000],
    visualize={use_plotly},
    repeat=1,
    mu={mu},
    task={pipeline["opt_task"]!r},
    source={pipeline["source"]!r},
    contact_target_method={pipeline["contact_target_method"]!r},
    contact_threshold={contact_threshold},
    sdf_backend={pipeline["sdf_backend"]!r},
    sdf_query_fn=sdf_query_fn,
    sdf_prepare_fn=sdf_prepare_fn,
)
if {opt_render!r} == "pyrender":
    print("[demo] pyrender backend is not available in optimisation code yet; falling back to plotly rendering.")

if {pipeline["opt_task"]!r} == "GH":
    opt.run_idx_json({sample_idx})
else:
    opt.run_idx_json_nv({sample_idx})
"""


def build_sim_code(robot_name, sample_idx, mu, exp_name, sim_render, sim_record, test_sdf):
    return f"""
from simulation.environment import Environment

env = Environment(
    robot={robot_name!r},
    mu={mu},
    exp={exp_name!r},
    repeat=1,
    render={sim_render!r},
    data_sample=None,
    task="test",
    test_sdf={test_sdf},
)
env.run_idx({sample_idx}, record={sim_record})
"""


def result_json_path(repo_root, robot, mu, sample_idx, source, task):
    if task == "GH" and source == "ycb":
        folder = "results"
    elif task == "GH" and source == "sdf":
        folder = "results_sdf"
    elif task == "NV" and source == "ycb":
        folder = "results_nv"
    else:
        folder = "results_nv_sdf"
    return (
        repo_root
        / "optimisation"
        / folder
        / robot
        / f"mu{mu:.1f}"
        / f"{sample_idx}_{robot}_0_mu_{mu:.1f}.json"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run demo with structured options: SDF source -> method -> clustering."
    )
    parser.add_argument(
        "--sdf-source",
        choices=["gt_pv", "recon_pv"],
        required=True,
        help="First-level option: where SDF comes from.",
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "genhand"],
        required=True,
        help="Second-level option: optimisation method.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "hdbscan", "mano"],
        default=None,
        help="Third-level option (required for genhand): contact clustering.",
    )
    parser.add_argument("--sample-idx", type=int, default=70, help="Index in full test split.")
    parser.add_argument("--robot", choices=["Shadow", "Allegro", "Barrett", "Robotiq"], default="Shadow")
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--device", default="cpu", help="Optimization device, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--network-ckpt",
        default=str((Path(__file__).resolve().parents[1] / "network" / "ckpt" / "snapshot_1599.pth.tar")),
        help="Checkpoint path used in stage-1 network inference/reconstruction.",
    )
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--opt-render", choices=["none", "plotly"], default="plotly")
    parser.add_argument("--sim-render", choices=["DIRECT", "GUI"], default="DIRECT")
    parser.add_argument("--sim-record", action="store_true")
    args = parser.parse_args()

    if args.method == "genhand" and args.cluster_method is None:
        parser.error("--cluster-method is required when --method genhand.")
    if args.method == "baseline" and args.cluster_method is not None:
        print("[demo] --cluster-method is ignored for method=baseline.")
    if args.method == "baseline":
        args.method = "nv"

    pipeline = resolve_pipeline(args.sdf_source, args.method, args.cluster_method)

    repo_root = Path(__file__).resolve().parents[1]
    network_dir = repo_root / "network"
    py_exec = sys.executable

    if pipeline["run_network_stage"]:
        print("[stage 1/3] network inference + reconstruction")
        run_cmd(
            [
                py_exec,
                "test_single.py",
                "--sample-idx",
                str(args.sample_idx),
                "--model-path",
                str(args.network_ckpt),
                "--recon-mode",
                pipeline["recon_mode"],
                "--device",
                args.device,
            ],
            cwd=str(network_dir),
        )
    else:
        print("[stage 1/3] network inference skipped (not required for selected SDF source)")

    print("[stage 2/3] grasp optimization")
    opt_code = build_opt_code(
        args.robot,
        args.sample_idx,
        args.device,
        args.opt_render,
        args.mu,
        pipeline,
        args.contact_threshold,
    )
    run_cmd([py_exec, "-c", opt_code], cwd=str(repo_root))

    result_path = result_json_path(
        repo_root=repo_root,
        robot=args.robot,
        mu=args.mu,
        sample_idx=args.sample_idx,
        source=pipeline["source"],
        task=pipeline["opt_task"],
    )
    sim_allowed = True
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            if isinstance(result_data, dict) and "error" in result_data:
                print(f"[demo] optimisation failed: {result_data['error']}")
                print("[demo] simulation is skipped because optimisation result is invalid.")
                sim_allowed = False
        except Exception as exc:
            print(f"[demo] warning: cannot parse optimisation result json: {exc}")

    if sim_allowed:
        print("[stage 3/3] simulation check")
        sim_code = build_sim_code(
            args.robot,
            args.sample_idx,
            args.mu,
            pipeline["sim_exp"],
            args.sim_render,
            args.sim_record,
            pipeline["source"] == "sdf",
        )
        run_cmd([py_exec, "-c", sim_code], cwd=str(repo_root))

    print("[done] demo pipeline completed.")


if __name__ == "__main__":
    main()
