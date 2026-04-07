import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class LiveViewNode(Node):
    def __init__(self):
        super().__init__('live_view_node')
        
        # Subscribe to the /in topic
        # Depth 10 handles the 'bunch of images' stream efficiently
        self.subscription = self.create_subscription(
            Image,
            '/in',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        self.get_logger().info('Live view started. Subscribed to /in')

    def listener_callback(self, msg):
        try:
            # Convert ROS message to OpenCV (BGR for display)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Open a window and show the frame
            cv2.imshow("ROS2 Live Stream: /in", frame)
            
            # waitKey(1) is required to actually render the window
            # If you press 'q' while the window is focused, it will close
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Closing viewer...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LiveViewNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
