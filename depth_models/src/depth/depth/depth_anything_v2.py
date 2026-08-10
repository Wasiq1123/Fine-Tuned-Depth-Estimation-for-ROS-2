#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append('/home/wasiq/testing_model/depth_models/src/Depth-Anything-V2')
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class DAv2Node(Node):
    def __init__(self):
        super().__init__('dav2_node')

        # Params 
        # frame_skip: run inference on 1 out of every N incoming RGB frames.

        self.declare_parameter('frame_skip', 1)
        self.frame_skip = int(self.get_parameter('frame_skip').value)

        # model_key: which FINE-TUNED checkpoint to load, by dataset name
        
        self.declare_parameter('model_key', 'scannet')
        model_key = self.get_parameter('model_key').value

        self.rgb_counter = 0

        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 1)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw_cal', 10)

        self.bridge = CvBridge()
        self.infer_rgb_transform = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

        self.get_logger().info("Loading DAv2 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model architecture configs (encoder backbone shapes)
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        # Fine-tuned checkpoints only, keyed by dataset name 
        # Both checkpoints are vits.
        self.model_list = {
            'scannet': "/home/wasiq/Downloads/save_models/save_models/20_epochs_scannet_documenatation_model.pth",
            'nyuv2': "/home/wasiq/Downloads/save_models/save_models/early_stopping_56_out_of_60_epochs_multi_loss_documenatation_model.pth",
        }
        self.model_encoder_map = {
            'scannet': 'vits',
            'nyuv2': 'vits',
        }

        if model_key not in self.model_list:
            available = ', '.join(sorted(self.model_list.keys()))
            raise ValueError(f"Unknown model_key='{model_key}'. Available keys: {available}.")

        self.encoder = self.model_encoder_map[model_key]
        checkpoint_path = self.model_list[model_key]

        self.model = DepthAnythingV2(**self.model_configs[self.encoder]).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)


        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.get_logger().info("Loaded checkpoint via 'model_state_dict' key.")
        else:
            self.model.load_state_dict(checkpoint)
            self.get_logger().info("Loaded checkpoint as raw state_dict.")

        self.model.eval()

        self.get_logger().info(
            f"Model loaded successfully on {self.device} ({checkpoint_path}, encoder={self.encoder}). "
            f"frame_skip={self.frame_skip} (running inference on 1 of every "
            f"{self.frame_skip} RGB frames)")

    def rgb_callback(self, msg):
        run_inference = (self.rgb_counter % self.frame_skip == 0)
        self.rgb_counter += 1
        if not run_inference:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        input_tensor = self.infer_rgb_transform(image=cv_image)["image"].unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            depth = self.model(input_tensor)
        inference_time = time.perf_counter() - start
        self.get_logger().info(f"Time required: {inference_time:.4f}")

        depth = depth.squeeze().cpu().numpy()
        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

        self.get_logger().info("Message has been published")


def main():
    rclpy.init()
    node = DAv2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()