#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import time
from ultralytics import YOLO


class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')

        # Only run inference on 1 out of every N incoming RGB frames.
        # Set to 1 to run inference on every frame.

        self.declare_parameter('frame_skip', 1)
        self.frame_skip = int(self.get_parameter('frame_skip').value)
        self.rgb_counter = 0  # counts every RGB frame received, skipped or not

        # Fine-tuned checkpoints only 
 
        self.model_paths = {
            '150epochs': '/home/wasiq/Downloads/save_models/yolo_depth/150_epochs_yolo_results.pt',
            'scannet_batch2': '/home/wasiq/Downloads/save_models/yolo_depth/scannet_batch2.pt',
        }

        self.declare_parameter('model_key', 'scannet_batch2')
        model_key = self.get_parameter('model_key').value

        if model_key not in self.model_paths:
            available = ', '.join(self.model_paths.keys())
            raise ValueError(f"Unknown model_key='{model_key}'. Available keys: {available}.")

        self.weights_path = self.model_paths[model_key]

        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 1)

        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw_cal', 10)

        self.bridge = CvBridge()

        self.get_logger().info(f"Loading YOLO26-depth fine-tuned model: {self.weights_path}")
        self.model = YOLO(self.weights_path)
        self.get_logger().info(
            f"Model loaded successfully={self.weights_path}. frame_skip={self.frame_skip} "
            f"(running inference on 1 of every {self.frame_skip} RGB frames)")

    def rgb_callback(self, msg):
        run_inference = (self.rgb_counter % self.frame_skip == 0)
        self.rgb_counter += 1
        if not run_inference:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        t0 = time.perf_counter()
        results = self.model(cv_image, verbose=False)
        infer_dt = time.perf_counter() - t0
        result = results[0]

        if result.depth is None:
            self.get_logger().warning("Model returned no depth output for this frame — skipping.")
            return

        depth = result.depth.data.cpu().numpy()
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            self.get_logger().warning(f"Unexpected depth shape {depth.shape} — skipping frame.")
            return

        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)

        self.get_logger().info(f"Depth frame published (inference {infer_dt:.4f}s)")


def main():
    rclpy.init()
    node = YoloDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()