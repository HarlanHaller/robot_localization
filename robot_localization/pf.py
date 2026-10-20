#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import os

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from logic_provider import LogicProvider, Particle
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion, PoseStamped
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper, draw_random_sample
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler
import random
from scipy.stats import truncnorm
from ament_index_python.packages import get_package_share_directory

class ParticleFilter(Node):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            last_scan_timestamp: this is used to keep track of the clock when using bags
            scan_to_process: the scan that our run_loop should process next
            occupancy_field: this helper class allows you to query the map for distance to closest obstacle
            transform_helper: this helps with various transform operations (abstracting away the tf2 module)
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            thread: this thread runs your main loop
    """
    def __init__(self):
        super().__init__('pf')
        self.base_frame = "base_link"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 

        self.n_particles = 1000          # the number of particles to use

        self.d_thresh = 0.2             # the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # the amount of angular movement before performing an update

        # TODO: define additional constants if needed

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)
        
        self.robot_pose_testing_pub = self.create_publisher(PoseStamped, "robot_pose_debug", 10)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud: list[Particle] = []

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)
        
        self.declare_parameter("n_particles_uniform", math.floor(self.n_particles*0.15))
        #We will use every n scan points starting from 0 where n is this paramater. (for optimization)
        self.declare_parameter("scan_point_density", 2)
        
        yaml_path = os.path.join(get_package_share_directory("robot_localization"), "config", "config.yaml")
        
        self.logic_provider = LogicProvider(self.occupancy_field, yaml_path) # pyright: ignore[reportArgumentType]

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)
        

    def pub_latest_transform(self):
        """ This function takes care of sending out the map to odom transform """
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.
            
            You do not need to modify this function, but it is helpful to understand it.
        """
        if self.scan_to_process is None:
            return
        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return
        
        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        # print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
        elif not self.particle_cloud:
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            # we have moved far enough to do an update!
            self.resample_particles()               # resample particles to focus on areas of high density
            self.update_particles_with_odom()    # update based on odometry
            self.update_particles_with_laser(r, theta)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose based on particles
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg.header.stamp)

    def moved_far_enough_to_update(self, new_odom_xy_theta):
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh


    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)
        """
        self.get_logger().info(" --- Update Robot Pose --- ")
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        # Chose the robot's pose based on the pose of the highest weighted particle.
        robot_pose = self.logic_provider.update_robot_pose(self.particle_cloud)
        
        if robot_pose is None:
            self.get_logger().warn("Can't choose best particle!")
            return
        
        self.robot_pose = robot_pose

        # Update the map offset based on robot_pose.
        if hasattr(self, 'odom_pose'):
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
             
            msg = PoseStamped()
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            msg.header = header
            msg.pose = self.robot_pose
            self.robot_pose_testing_pub.publish(msg)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self):
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a tuple (x,y,theta)
            that indicates the change in position and angle between the odometry
            when the particles were last updated and the current odometry.
        """
        self.get_logger().info(" --- Update Particle Locations --- ")
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return
        
        self.particle_cloud = self.logic_provider.update_particles_with_odom(delta, old_odom_xy_theta, self.particle_cloud) # pyright: ignore[reportArgumentType]

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample in helper_functions.py.
        """
        # make sure the distribution is normalized
        self.get_logger().info(" --- Resample Particle --- ")
        self.normalize_particles()
        
        particles = self.logic_provider.resample_particles(self.particle_cloud)
            
        self.particle_cloud = particles
        
        self.normalize_particles()

    def update_particles_with_laser(self, r, theta):
        """
        Updates the particle weights in response to the scan data.
        Each weight is higher as the difference between the closest measured
        distance from the laser scan and the distance from each particle
        to the nearest object to it.

        Args:
            r: the distance readings to obstacles
            theta: the angle relative to the robot frame
            for each corresponding reading.
        """
        self.get_logger().info(" --- Update Particle Weights --- ")
        # Calculate all the weights
        density = self.get_parameter("scan_point_density").value
        reduced_r = r[0:360:density]
        reduced_theta = theta[0:360:density]
        particles = self.logic_provider.update_particles_with_laser(reduced_r, reduced_theta, self.particle_cloud)
        self.particle_cloud = particles
        
        # if count_OOB >= 50:
        #     self.get_logger().warn(f"Number of out of bounds particles is: {count_OOB}")

        # Normalize the particle weights
        self.normalize_particles()

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is omitted, the odometry will be used """
                      
        self.get_logger().info(" --- Initial Particle Cloud --- ")
        
        if xy_theta is None:
            # If we dont get an initial guess then we just uniformly place guesses over the bounds of the map
            # xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose);
            pass
        self.particle_cloud = []
        
        particles = self.logic_provider.initialize_particle_cloud(timestamp, xy_theta)
        self.particle_cloud = particles
                
        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        self.particle_cloud = self.logic_provider.normalize(self.particle_cloud)

    def publish_particles(self, timestamp):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud[::5]:
            # print(p.w)
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w*1000)) # pyright: ignore[reportAttributeAccessIssue]
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop 
        if self.scan_to_process is None:
            self.scan_to_process = msg

def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
