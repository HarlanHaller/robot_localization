import rclpy
from rclpy.node import Node
from nav2_msgs.msg import ParticleCloud
from rclpy.qos import qos_profile_sensor_data

import matplotlib.pyplot as plt

class DebugInfo(Node):
    """This Node will listen on the /particle_cloud and plot a histogram of the recived weights. We could not 
    get this node to found by ROS so run it as a python file, it works fine. This Node should 
    be spun once to not constantly overwrite the figure. This Node is used as a unformal test to ensure
    all weights are in a resonable distobution.
    """
    def __init__(self):
        """Creates a DebugInfo Node
        """
        super().__init__("debug_node")
        self.sub = self.create_subscription(ParticleCloud, 'particle_cloud', self.process_data, qos_profile_sensor_data)
        
    def process_data(self, msg: ParticleCloud):
        """Takes msg input and prints the header of that message."""
        weights = []
        for _, p in enumerate(msg.particles):
            if p.weight > 0:
                weights.append(p.weight)
        # Make a simple histogram of weights
        plt.hist(weights, 25)
        plt.title("Histogram of Weights")
        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.savefig("./plots/current_weights.png")
        print("made figure")

def main(args=None):
    rclpy.init(args=args)           # Initialize communication with ROS
    node = DebugInfo()              # Create our Node
    rclpy.spin_once(node)           # Run the Node once
    rclpy.shutdown()                # cleanup

if __name__ == '__main__':
    main()