import math
from typing import List, Tuple
import numpy as np
from robot_localization.occupancy_field import OccupancyField
from robot_localization.helper_functions import TFHelper, draw_random_sample
from robot_localization.angle_helpers import quaternion_from_euler
import random
from scipy.stats import truncnorm, norm
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped
from rclpy.parameter import Parameter

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)

        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """
        Construct a new Particle

        Args:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
        """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))
        
    def as_tuple(self):
        """ A helper function to convert a particle to a (x, y, theta) Tuple"""
        return(self.x, self.y, self.theta)


class LogicProvider:
    def __init__(
        self,
        occupancy_field: OccupancyField,
        n_particles: int,
        n_particles_uniform_param: Parameter
    ):
        self.occupancy_field: OccupancyField = occupancy_field
        self.n_particles = n_particles
        self.n_particles_uniform_param = n_particles_uniform_param

    def update_robot_pose(self, particle_cloud: List[Particle]) -> Pose | None:
        best_weight = 0.0
        best_particle = None
        # weights = [p.w for p in self.particle_cloud];
        # print(max(weights))
        for particle in particle_cloud:
            if particle.w > best_weight:
                best_particle = particle
        if best_particle is None:
            return

        # Set robot_pose based on the pose of the highest weighted particle.
        robot_pose = Pose()
        robot_pose.position.x = best_particle.x
        robot_pose.position.y = best_particle.y
        quaternion = quaternion_from_euler(0, 0, best_particle.theta)
        robot_pose.orientation.x = quaternion[0]
        robot_pose.orientation.y = quaternion[1]
        robot_pose.orientation.z = quaternion[2]
        robot_pose.orientation.w = quaternion[3]

        return robot_pose

    def update_particles_with_odom(
        self,
        delta: Tuple,
        old_odom_xy_theta: Tuple,
        particle_cloud: List[Particle]
    ) -> List[Particle]:
        for p in particle_cloud:
            dT = p.theta - old_odom_xy_theta[2]
            new_x = p.x + (delta[0]*math.cos(dT) - delta[1]*math.sin(dT))
            new_y = p.y + (delta[0]*math.sin(dT) + delta[1]*math.cos(dT))
            new_t = p.theta + delta[2]
            p.x, p.y, p.theta = new_x, new_y, new_t
        return particle_cloud

    def resample_particles(self, particle_cloud: List[Particle]) -> List[Particle]:
        n_particles_uniform = self.n_particles_uniform_param.value
        prob_list = [p.w for p in particle_cloud]
        particles: list[Particle] = draw_random_sample(
            particle_cloud, prob_list, self.n_particles-n_particles_uniform)
        sd = 0.2 # noise level
        theta_spread = 4
        bounds = self.occupancy_field.get_obstacle_bounding_box()

        for p in particles:
            noise_x = random.gauss(0, sd)
            noise_y = random.gauss(0, sd)
            t_val = random.vonmisesvariate(p.theta, theta_spread)
            p.x, p.y, p.theta = p.x+noise_x, p.y+noise_y, t_val
        for _ in range(n_particles_uniform):
            x_pos = random.uniform(bounds[0][0], bounds[0][1])
            y_pos = random.uniform(bounds[1][0], bounds[1][1])
            theta_val = random.uniform(0, 2*math.pi)
            p = Particle(x = x_pos, y = y_pos, theta = theta_val)
            particles.append(p)

        return particles

    def update_particles_with_laser(
        self, r, theta, particle_cloud: List[Particle]
    ) -> Tuple[List[Particle], int]:
        """
        Calculate all the weights of particle cloud.
        """
        # Calculate the closest distance to the real robot.
        shortest_measured = min(r)
        # Count how many particles are out of bounds of the map.
        count_OOB = 0

        # For each particle:
        for particle in particle_cloud:
            # Calculate the distance to the nearest obstacle on the map
            shortest_calculated = (
                self.occupancy_field.get_closest_obstacle_distance(
                particle.x, particle.y))

            # If it is outside the bounds of the map:
            if math.isnan(shortest_calculated):
                # Set the weight to zero.
                particle.w = 0
                count_OOB += 1
                continue

            # Otherwise set the weight based on a normal
            # distribution around the real closest distance.
            weight = norm.pdf(shortest_calculated, loc=shortest_measured, scale=1)
            particle.w = weight

        return (particle_cloud, count_OOB)

    def initialize_particle_cloud(self, timestamp, xy_theta=None) -> List[Particle]:
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        x_div = 0.5 #kinda just a magic number should be tweaked 
        y_div = 0.5 #kinda just a magic number should be tweaked 
        # print(bounds);
        # print(xy_theta);
        # print(x_min_max);
        # print(y_min_max);
        theta_div = 4 #kinda just a magic number should be tweaked 
        particle_cloud: List[Particle] = []

        if xy_theta is not None:
            print("guess recived")
            x_min_max = ((bounds[0][0] - xy_theta[0])/x_div, (bounds[0][1] - xy_theta[0])/x_div)
            y_min_max = ((bounds[1][0] - xy_theta[1])/y_div, (bounds[1][1] - xy_theta[1])/y_div)
            x_vals = truncnorm.rvs(
                x_min_max[0], x_min_max[1], loc=xy_theta[0], scale=x_div, size=self.n_particles)
            y_vals = truncnorm.rvs(
                y_min_max[0], y_min_max[1], loc=xy_theta[1], scale=y_div, size=self.n_particles)
            for i in range(self.n_particles):
                x = x_vals[i]
                y = y_vals[i]
                t = random.vonmisesvariate(xy_theta[2], theta_div)
                p = Particle(x = x, y = y, theta = t)
                particle_cloud.append(p)
        else:
            for i in range(self.n_particles):
                x_pos = random.uniform(bounds[0][0], bounds[0][1])
                y_pos = random.uniform(bounds[1][0], bounds[1][1])
                theta_val = random.uniform(0, 2*math.pi)
                p = Particle(x = x_pos, y = y_pos, theta = theta_val)
                particle_cloud.append(p)

        return particle_cloud

    def normalize(self, particle_cloud):
        """
        Normalize the weights of a particle cloud so the sum equals one.

        Args:
            particle_cloud: A list of Particles to normalize.
        
        Returns:
            particle_cloud: A list of particles with normalized weights.
        """
        weight_sum = sum(particle.w for particle in particle_cloud)
        if abs(weight_sum - 1) < 1e-6:
            return
        for particle in particle_cloud:
            particle.w = particle.w / weight_sum

        return particle_cloud
