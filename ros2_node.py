"""
ros2_node.py - ROS2 node that subscribes to a camera topic and runs
paver detection on every incoming frame.

Run with:
    ros2 run <your_package> ros2_node

Or directly:
    python ros2_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import time

from detection import process_frame
from visualization import draw_paver


class PaverDetectionNode(Node):
    def __init__(self):
        super().__init__('paver_detection_node')

        # --- Parameters (tunable via ros2 param set at runtime) ---
        self.declare_parameter('image_topic', '/img')
        self.declare_parameter('show_window', True)
        self.declare_parameter('save_snapshots', False)

        topic        = self.get_parameter('image_topic').get_parameter_value().string_value
        self.show_window      = self.get_parameter('show_window').get_parameter_value().bool_value
        self.save_snapshots   = self.get_parameter('save_snapshots').get_parameter_value().bool_value

        self.subscription = self.create_subscription(
            Image,
            topic,
            self._image_callback,
            10,
        )

        self.bridge = CvBridge()
        self._prev_time = time.time()
        self._frame_count = 0

        self.get_logger().info(f'Paver detection node started. Subscribed to {topic}')

    # ------------------------------------------------------------------
    def _image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        # --- Run detection pipeline ---
        pavers, _ = process_frame(frame)

        # --- Draw results ---
        for pose in pavers:
            draw_paver(
                frame,
                cnt=pose['contour'],
                centroid=pose['centroid'],
                unique_corner=pose['unique_corner'],
                orientation_angle=pose['orientation_angle'],
                approx_pts=pose['approx_pts'],
            )
            self.get_logger().debug(
                f"Paver @ {pose['centroid']}  angle={pose['orientation_angle']:.2f} deg"
            )

        # --- FPS ---
        now = time.time()
        fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

        

        # --- Log paver count at INFO level (not every frame — throttled) ---
        if self._frame_count % 30 == 0:
            self.get_logger().info(f'Pavers detected: {len(pavers)}')

        # --- Display window ---
        if self.show_window:
            cv.imshow('Paver Detection (ROS2)', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit key pressed — shutting down.')
                rclpy.shutdown()
            elif key == ord('s') or self.save_snapshots:
                filename = f'snapshot_{self._frame_count:06d}.jpg'
                cv.imwrite(filename, frame)
                self.get_logger().info(f'Saved {filename}')

        self._frame_count += 1


# ------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PaverDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()