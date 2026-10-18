import rclpy
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud

import matplotlib.pyplot as plt

class DebugInfo(Node):
    def __init__(self):
        super().__init__("debug_node")
        self.sub = self.create_subscription(ParticleCloud, 'particle_cloud', self.process_data, 10)
        
    def process_data(self, msg: ParticleCloud):
        """Takes msg input and prints the header of that message."""
        weights = []
        for _, p in enumerate(msg.particles):
            weights.append(p.weight)
        plt.hist(weights, 25)
        plt.title("histogram of weights")

def main(args=None):
    rclpy.init(args=args)           # Initialize communication with ROS
    node = DebugInfo()              # Create our Node
    rclpy.spin_once(node)           # Run the Node once
    rclpy.shutdown()                # cleanup

if __name__ == '__main__':
    main()