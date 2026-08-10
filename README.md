# depth — Multi-Model Monocular Depth Estimation for ROS 2

A ROS 2 (Humble) package that wraps two monocular depth estimation models, YOLO26 Depth and Depth Anything V2, as interchangeable nodes, using checkpoints fine-tuned in-house rather than the stock base weights. Both share the same RGB input topic and depth output topic, so you can drop either one into a robot's perception stack, or run them side by side to compare.

## Contents

- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Fine-Tuning](#fine-tuning)
- [Package Structure](#package-structure)
- [Setup](#setup)
- [Topics](#topics)
- [Running the Nodes](#running-the-nodes)
- [Model Comparison](#model-comparison)
- [Testing / Evaluation](#testing--evaluation)
- [License](#license)

## Overview

This package has two separate ROS 2 nodes. Each one subscribes to a live RGB image stream and publishes a metric depth prediction.

`yolo_depth` runs Ultralytics' YOLO26 Depth architecture, loaded with our own fine-tuned checkpoints rather than the stock pretrained weights. `depth_anything_v2` runs Depth Anything V2 in metric depth mode, also loaded with our own fine-tuned checkpoints.

Both nodes expose the checkpoint choice and frame skip rate as ROS 2 parameters, so you can swap checkpoints or throttle inference load at launch time without touching the code.

## Models Implemented

| Node | Model | Notes |
|---|---|---|
| `yolo_depth` | [YOLO26 Depth](https://docs.ultralytics.com/tasks/depth) | Fast, single shot depth head from Ultralytics, fine-tuned checkpoints |
| `depth_anything_v2` | [Depth Anything V2 (metric depth)](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) | ViT based, strong accuracy, heavier compute, fine-tuned checkpoints |

## Fine-Tuning

Both models were fine-tuned in two stages rather than used off the shelf:

1. **Stage 1 — NYU Depth V2.** Both YOLO26 Depth and Depth Anything V2 were fine-tuned on the NYU Depth V2 dataset. DAv2 was trained for 60 epochs (checkpoint: `early_stopping_56_out_of_60_epochs_multi_loss_documenatation_model.pth`, early-stopped at epoch 56). YOLO26 Depth was trained for 150 epochs (checkpoint: `150_epochs_yolo_results.pt`).
2. **Stage 2 — ScanNet.** The stage 1 (NYU Depth V2 fine-tuned) checkpoints were then further fine-tuned on ScanNet, producing the ScanNet-stage checkpoints referenced in the node parameters below.

For full training configuration, data splits, and fine-tuning methodology, see the separate repo: **fine-tuning of depth models**.

## Package Structure

```
depth/
├── depth/
│   ├── yolo_depth.py          # YOLO26 Depth ROS 2 node
│   ├── depth_anything_v2.py   # DAv2 ROS 2 node
├── package.xml
├── setup.py
└── setup.cfg
```

## Setup

Clone this package into your ROS 2 workspace `src/` folder and build it with `colcon build` once the model dependencies below are installed.

### 1. Depth Anything V2 (DAv2)

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt
```

This node does **not** use the base DAv2 checkpoints from the official repo it loads our own fine-tuned checkpoints (see [Fine-Tuning](#fine-tuning)). Download the fine-tuned checkpoints and update the checkpoint paths in `depth/depth_anything_v2.py` (`self.model_list`) to point at wherever you saved them.

If you're running on CPU, uninstall `xformers`. It's a GPU only dependency and will throw an error otherwise. See [DepthAnything/Depth-Anything-V2#312](https://github.com/DepthAnything/Depth-Anything-V2/issues/312) for the exact error.

```bash
python3 -m pip uninstall xformers
```

### 2. YOLO26 Depth

```bash
pip install ultralytics
```

This node does **not** use the base YOLO26 Depth weights it loads our own fine-tuned checkpoints (see [Fine-Tuning](#fine-tuning)). Download the fine-tuned checkpoints and update the checkpoint paths in `depth/yolo_depth.py` (`self.model_paths`) to point at wherever you saved them.

See the official task docs for background on the base architecture: [Ultralytics — Depth Estimation](https://docs.ultralytics.com/tasks/depth).

## Topics

Both nodes share the same topic interface.

| Direction | Topic | Type |
|---|---|---|
| Subscribed | `/camera/color/image_raw` | `sensor_msgs/Image` |
| Published | `/camera/depth/image_raw_cal` | `sensor_msgs/Image` (`32FC1`) |

## Running the Nodes

### Depth Anything V2

```bash
ros2 run depth depth_anything_v2 --ros-args -p model_key:=scannet -p frame_skip:=3
```

```bash
ros2 run depth depth_anything_v2 --ros-args -p model_key:=nyuv2 -p frame_skip:=3
```

`model_key` options:

| Key | Checkpoint | Fine-tuning stage |
|---|---|---|
| `nyuv2` | `early_stopping_56_out_of_60_epochs_multi_loss_documenatation_model.pth` | Stage 1: NYU Depth V2 only |
| `scannet` | `20_epochs_scannet_documenatation_model.pth` | Stage 2: NYU Depth V2 → ScanNet |

### YOLO26 Depth

```bash
ros2 run depth yolo_depth --ros-args -p model_key:=150epochs -p frame_skip:=3
```

```bash
ros2 run depth yolo_depth --ros-args -p model_key:=scannet_batch2 -p frame_skip:=3
```

`model_key` options:

| Key | Checkpoint | Fine-tuning stage |
|---|---|---|
| `150epochs` | `150_epochs_yolo_results.pt` | Stage 1: NYU Depth V2 only |
| `scannet_batch2` | `scannet_batch2.pt` | Stage 2: NYU Depth V2 → ScanNet |

`frame_skip` works the same way across both nodes and takes any positive integer. It controls how many incoming RGB frames go by per inference run, so `frame_skip:=3` runs inference on 1 out of every 3 frames.

> Running both nodes at the same time: both default to overlapping node/topic names (see [Package Structure](#package-structure) and [Topics](#topics)). Remap `__node` and the output topic per instance if you need them running concurrently rather than sequentially — this hasn't been verified against your specific ROS 2 setup, so confirm behavior before relying on it.

## Model Comparison

Evaluated with the same rosbag/Gazebo ground-truth pipeline described in [Testing / Evaluation](#testing--evaluation). Lower is better for AbsRel, RMSE, and LogRMSE. Higher is better for δ1 through δ3.

### Depth Anything V2 — base vs. fine-tuned stages

| Model | AbsRel↓ | RMSE↓ | LogRMSE↓ | δ1↑ | δ2↑ | δ3↑ |
|---|---|---|---|---|---|---|
| DAv2 Base | 1.245 | 2.286 | 0.779 | 0.116 | 0.221 | 0.407 |
| Early Stop 56/60 (`nyuv2`) | 1.213 | 1.963 | 0.796 | 0.112 | 0.238 | 0.413 |
| ScanNet 20 Epochs (`scannet`) | **0.986** | **1.912** | **0.699** | **0.144** | **0.333** | **0.508** |

**ScanNet 20-epoch fine-tune vs. DAv2 base:**

| Metric | Change |
|---|---|
| AbsRel | **20.8% reduction** |
| RMSE | 16.4% reduction |
| LogRMSE | 10.2% reduction |
| Delta1 (δ<1.25) | +24.0% |
| Delta2 (δ<1.25²) | **+51.1%** |
| Delta3 (δ<1.25³) | +24.9% |

### YOLO26 Depth — base vs. fine-tuned stages

| Model | Frames | AbsRel↓ | RMSE↓ | LogRMSE↓ | Delta1↑ | Delta2↑ | Delta3↑ |
|---|---|---|---|---|---|---|---|
| YOLO Base Model | 279 | 2.4376 | 4.0132 | 1.2022 | 0.0195 | 0.0562 | 0.1731 |
| YOLO 150 Epochs (`150epochs`) | 279 | 1.2619 | 2.0480 | 0.8134 | 0.1107 | 0.3359 | 0.5050 |
| ScanNet Batch 1 | 278 | 0.5673 | 2.1515 | 0.6380 | 0.2085 | 0.4435 | 0.7218 |
| ScanNet Batch 2 (`scannet_batch2`) | 279 | **0.5047** | 2.1524 | **0.6162** | **0.2239** | **0.5053** | **0.7637** |

> ScanNet Batch 1 is included above for completeness but is no longer exposed as a `model_key` option on the `yolo_depth` node — only `150epochs` and `scannet_batch2` are currently selectable (see [Running the Nodes](#running-the-nodes)).

**ScanNet Batch 2 vs. base model (best result):**

| Metric | Change |
|---|---|
| AbsRel | **79.3% reduction** |
| RMSE | 46.4% reduction |
| LogRMSE | 48.7% reduction |
| Delta1 (δ<1.25) | **+1049.6% (10.5x)** |
| Delta2 (δ<1.25²) | +799.8% (8x) |
| Delta3 (δ<1.25³) | +341.1% (4.4x) |


### Best YOLO checkpoint vs. best DAv2 checkpoint

| Metric | Top YOLO | Top DAv2 | Winner |
|---|---|---|---|
| AbsRel↓ | **0.505** | 0.986 | YOLO |
| RMSE↓ | 2.370 | **1.912** | DAv2 |
| LogRMSE↓ | **0.641** | 0.699 | YOLO |
| Delta1↑ | **0.242** | 0.144 | YOLO |
| Delta2↑ | **0.506** | 0.333 | YOLO |
| Delta3↑ | **0.730** | 0.508 | YOLO |

## Testing / Evaluation

The accuracy numbers previously in this README came from a custom ROS 2 bag based evaluation pipeline. RGB and Gazebo ground truth depth are matched with `message_filters.ApproximateTimeSynchronizer`, and per frame AbsRel, RMSE, LogRMSE, and δ1 through δ3 get logged to CSV for offline aggregation. For a general reference on structuring depth model testing in ROS 2, the ROS 2 [`message_filters`](https://github.com/ros2/message_filters) package docs cover the synchronization approach this is built on.

## License

Apache-2.0
