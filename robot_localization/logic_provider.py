import math
from typing import LiteralString
from typing import List, Tuple
import numpy as np
import yaml
from robot_localization.occupancy_field import OccupancyField
from robot_localization.helper_functions import TFHelper, draw_random_sample
from robot_localization.angle_helpers import quaternion_from_euler
import random
from scipy.stats import truncnorm, norm
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped
from rclpy.parameter import Parameter
import matplotlib.pyplot as plt
import time

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
        config_path: LiteralString
    ):
        self.occupancy_field: OccupancyField = occupancy_field
        self.config_path = config_path
        
        #Yaml file setup:
        # n_particles: int
        # percent_particles_uniform: float
        # resample_noise_sd: float
        # resample_theta_spread: int
        # laser_gaussian_sd: float
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            self.n_particles = data["n_particles"]
            self.n_particles_uniform = math.floor(self.n_particles*data["percent_particles_uniform"])
            self.resample_noise_sd = data["resample_noise_sd"]
            self.resample_theta_spread = data["resample_theta_spread"]
            self.laser_gaussian_sd = data["laser_gaussian_sd"]
                    
    
    def update_robot_pose(self, particle_cloud: List[Particle]) -> Pose | None:
        """
        Estimate the pose of the robot based on
        the poses and weights of a particle cloud.

        Args:
            particle_cloud (list of Particles): The particles
            to use to estimate the pose of the robot.
        
        Returns:
            None: If none of the particles have weights above zero.
            robot_pose (Pose): The position and orientation of
            the robot pose estimate.
        """
        # Find the particle with the highest weight.
        best_weight = 0.0
        best_particle = None
        # weights = [p.w for p in self.particle_cloud];
        # print(max(weights))
        for particle in particle_cloud:
            if particle.w > best_weight:
                best_particle = particle
                best_weight = particle.w

        # If there are no particles with weights above zero:
        if best_particle is None:
            # Return None.
            return

        #print(max(p.w for p in particle_cloud) == best_particle.w)

        # Set robot_pose based on the pose of the highest weighted particle.
        # Convert from x, y and theta to a 3D position vector and a quaternion angle.
        robot_pose = Pose()
        robot_pose.position.x = best_particle.x
        robot_pose.position.y = best_particle.y
        quaternion = quaternion_from_euler(0, 0, best_particle.theta)
        robot_pose.orientation.x = quaternion[0]
        robot_pose.orientation.y = quaternion[1]
        robot_pose.orientation.z = quaternion[2]
        robot_pose.orientation.w = quaternion[3]

        # Return the estimated pose of the robot.
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
        n_particles_uniform: int = self.n_particles_uniform
        prob_list = [p.w for p in particle_cloud]
        particles: list[Particle] = draw_random_sample(
            particle_cloud, prob_list, self.n_particles-n_particles_uniform)
        sd = self.resample_noise_sd # noise level
        theta_spread = self.resample_theta_spread # paramater to vonmisesvariate, higher number = less variation
        bounds = self.occupancy_field.get_obstacle_bounding_box()

        for p in particles:
            noise_x = random.gauss(0, sd)
            noise_y = random.gauss(0, sd)
            t_val = random.vonmisesvariate(p.theta, theta_spread)
            # t_val = p.theta
            p.x, p.y, p.theta = p.x+noise_x, p.y+noise_y, t_val
        for _ in range(n_particles_uniform):
            x_pos = random.uniform(bounds[0][0], bounds[0][1])
            y_pos = random.uniform(bounds[1][0], bounds[1][1])
            theta_val = random.uniform(0, 2*math.pi)
            p = Particle(x = x_pos, y = y_pos, theta = theta_val, w=0.0)
            particles.append(p)

        # print("resample done")
        return particles

    @classmethod
    def gaussian_pdf(cls, points, std:float):
        """
        PDF from a zero-centered gaussian distribution.

        Args:
            points (list of floats): points to sample at
            std (float): two times the standard deviation
            of the gaussian squared
        
            Returns:
                A list of floats representing the values
                of the gaussian sampled at points
        """
        return np.exp(np.square(points) / -std)

    def update_particles_with_laser(
        self, r, theta, particle_cloud: List[Particle]
    ) -> List[Particle]:
        """
        Calculate all the weights of a particle cloud.

        Args:
            r (list of floats): The distances measurements from a
            LiDAR scanner, each index corresponding with a different angle.
            theta (list of floats): The angles of the LiDAR scan,
            each index lines up with r.
            particle_cloud (list of Particles): The particles to weight.
        """
        # Calculate the closest distance to the real robot.
        sd = self.laser_gaussian_sd
        sd_preprossed = sd*sd * 2
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        # print(r)
        # print(theta)
        #start_time = time.perf_counter()

        # Project the scan points into a matrix where each column
        # is a different scan point represented by the vector [x; y; 1]
        scan_points = np.zeros((3, len(r)))
        # For each point:
        for i in range(len(r)):
            # Unless the distance from the robot to the point is infinite:
            if math.isinf(r[i]):
                continue
            # Calculate the x and y coordinate of the point in the map frame
            scan_points[0, i] = math.cos(theta[i]) * r[i]
            scan_points[1, i] = math.sin(theta[i]) * r[i]
            scan_points[2, i] = 1

        # print(f"pre-compute: {time.perf_counter()-start_time}")

        # For each particle:
        for particle in particle_cloud:
            # If the particles is outside the bounds
            # of the map set their weight to zero.
            if(
                particle.x < bounds[0][0] or
                particle.x > bounds[0][1] or
                particle.y < bounds[1][0] or
                particle.y > bounds[1][1]
            ):
                particle.w = 0.0
                # print("OOB")
                continue

            # Otherwise:
            # Multiply each scan point vector by a matrix that
            # rotates and translates it from the robot frame
            # onto the particle in the map frame as if the particle
            # took the scan itself.
            t = particle.theta
            T = np.array([
                [math.cos(t), -math.sin(t), particle.x],
                [math.sin(t), math.cos(t),  particle.y],
                [0,           0,            1]
            ])
            particle_space = T @ scan_points
            # print(f"transform: {time.perf_counter()-start_time}")

            # Calculate the distance from each projected scan point
            # to the nearest obstacle on the map.
            dists = np.zeros(particle_space.shape[1])
            for i, cords in enumerate(particle_space.T):
                dists[i] = self.occupancy_field.get_closest_obstacle_distance(cords[0], cords[1])
                # If the returned value is out of bounds
                # assume it is very far away.
                if math.isnan(dists[i]):
                    dists[i] = 100

            # print(f"get dist: {time.perf_counter()-start_time}")

            # Scale each distance by plugging them into
            # a gaussian probability distribution function.
            confidence_sum = sum(self.gaussian_pdf(dists, sd_preprossed))

            # print(f"get weight: {time.perf_counter()-start_time}")

            # Divide the sum by the number of scan points to get the average weight.
            weight = confidence_sum / len(dists)
            # Set that average weight to be the weight of the particle.
            particle.w = weight * weight

        #count_OOB = 0 archive from debugging, left in for evidence
        # print(f"end: {time.perf_counter()-start_time}")

        # Return the particle cloud with updated weights.
        return particle_cloud

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
            x_vals: List[int] = truncnorm.rvs(
                x_min_max[0], x_min_max[1], loc=xy_theta[0], scale=x_div, size=self.n_particles) # pyright: ignore[reportAssignmentType]
            y_vals: List[int] = truncnorm.rvs(
                y_min_max[0], y_min_max[1], loc=xy_theta[1], scale=y_div, size=self.n_particles) # pyright: ignore[reportAssignmentType]
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

    def normalize(self, particle_cloud) -> List[Particle]:
        """
        Normalize the weights of a particle cloud so the sum equals one.

        Args:
            particle_cloud: A list of Particles to normalize.
        
        Returns:
            particle_cloud: A list of particles with normalized weights.
        """
        # Calculate the sum of the current weights.
        weight_sum = sum(particle.w for particle in particle_cloud)

        # If the particles are already normalized save time by not updating them.
        if abs(weight_sum - 1) < 1e-6:
            return particle_cloud

        # Divide each particle weight by the sum of all the original weights.
        for particle in particle_cloud:
            particle.w = particle.w / weight_sum

        # Return the updated particle cloud.
        return particle_cloud
