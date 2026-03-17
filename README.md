# GenHand 
---

The offical code base for [GenHand: generalised human grasp kinematic retargeting](https://www.nature.com/articles/s44182-026-00076-1).


### Repository modules:
- `dataset/`: data loading and preprocessing for DexYCB-based training/testing.
- `network/`: hand/object pose estimation and mesh reconstruction, alsoe contains modified [manopth](https://github.com/hassony2/manopth).
- `optimisation/`: grasp optimization with either ground truth input or reconstructed input.
- `simulation/`: PyBullet-based grasp execution and validation.
- `demo/`: a script to quickly run a retargeting demo
- `thirdparty`: contains modified [Pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) and [Pytorch_volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric)
### Repository Layout

```text
.
├── dataset/
├── network/
├── optimisation/
├── simulation/
├── demo/
├── urdf/
└── thirdparty/
    ├── pytorch_kinematics/
    └── pytorch_volumetric/
```

### Installation

Create a conda environment first (Python 3.10 recommended), then install dependencies and local third-party packages:

```bash
conda create -n genhand python=3.7 -y
conda activate genhand
pip install -r requirements.txt
pip install -e .
pip install -e network/manopth
pip install -e thirdparty/pytorch_kinematics
pip install -e thirdparty/pytorch_volumetric
```

### Data and Checkpoints

Expected local resources:
- network checkpoint: please download the pre-trained [checkpoint](https://drive.google.com/file/d/14j6HzB9KbVL2u4i-OAy_Ed9aauvmNnnW/view?usp=drive_link) and put it under `network/ckpt/`. 

- urdf file: please download the robot [urdf files](https://drive.google.com/file/d/1uwH9Sc5zGj5SaeRDERgrzD0MDKf5WEzO/view?usp=drive_link) and put it under the repo

### Run the Demo

Retargeting with ground truth meshes, the method can be either genhand or baseline, sample index is fixed to 70 as a sample before download and preprocess the whole dataset. The robot can be Shadow, Allegro, Barrett or Robotiq.

```bash
python3 demo/run_demo.py --sdf-source gt_pv --method genhand --cluster-method hdbscan --contact-threshold 0.05  --sample-idx 70 --robot Shadow --device cuda:0 --sim-render GUI
```


### Citation
```text
@article{qi2026genhand,
  title={GenHand: generalised human grasp kinematic retargeting},
  author={Qi, Liyuan and Popoola, Olaoluwa and Imran, Muhammad Ali and Ahmad, Wasim},
  journal={npj Robotics},
  volume={4},
  number={1},
  pages={19},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

### Acknowledgements
This repo is partially build upon [AlignSDF](https://github.com/zerchen/AlignSDF), [DFC](https://github.com/tengyu-liu/diverse-and-stable-grasp), [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp). Also shout out to [Pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics), [Pytorch_volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric), [manopth](https://github.com/hassony2/manopth) and [Dexycb](https://dex-ycb.github.io/) for the wonderful tools and data. Please also consider cite those works.


### TODO
This repo is actively maintained currently to improve readability, but might move slow due to the time and resource limits. Please raise a issue if you have any question.
