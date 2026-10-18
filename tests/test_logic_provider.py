import math
import os
import random
from typing import List
import pytest
import numpy as np
import yaml
from robot_localization.angle_helpers import euler_from_quaternion
from robot_localization.logic_provider import LogicProvider, Particle
from test_support_classes.support_occupancy_field import OccupancyField
from test_support_classes.support_parameter import Parameter


class TestBase:
    def create_logic_provider(self, n_particles = 300, n_particles_uniform = 50):
        self.occupancy_field = OccupancyField("mapdata/gauntlet.pkl")
        self.n_particles = n_particles
        self.n_particles_uniform_param = Parameter(n_particles_uniform)
        self.logic_provider = LogicProvider(
            self.occupancy_field, self.n_particles, self.n_particles_uniform_param)


class TestUpdateWithOdom(TestBase):
    def array_to_particles(self, arr) -> List[Particle]:
        particles: List[Particle] = []
        for i in range(arr.shape[1]):
            p = Particle(x = arr[0, i], y=arr[1,i], theta=arr[2,i])
            particles.append(p)
        return particles
    
    def particles_to_array(self, plist):
        arr = np.array([[p.x, p.y, p.theta] for p in plist]).T
        return arr
            
    
    
    def test_move_1p_1(self):
        self.create_logic_provider()
        
        particles = [Particle(x = 1, y = 2, theta=(3 * math.pi / 2))]
        N_start = (2, 1, math.pi / 2)
        N_end = (2, 2, math.pi / 6)
        d_x, d_y, d_t = N_end[0] - N_start[0], N_end[1] - N_start[1], N_end[2] - N_start[2]
        new_particles = self.logic_provider.update_particles_with_odom((d_x, d_y, d_t), N_start, particles)
        
        expected_theta = 3 * math.pi / 2 + d_t
        
        assert new_particles[0].as_tuple() == pytest.approx((1, 1, expected_theta))
        
    def test_move_1p_2(self):
        self.create_logic_provider()
        
        particles = [Particle(x = 1, y = 2, theta=(0))]
        N_start = (2, 1, math.pi / 2)
        N_end = (2, 2, math.pi / 6)
        d_x, d_y, d_t = N_end[0] - N_start[0], N_end[1] - N_start[1], N_end[2] - N_start[2]
        new_particles = self.logic_provider.update_particles_with_odom((d_x, d_y, d_t), N_start, particles)
        
        expected_theta = 0 + d_t
        
        assert new_particles[0].as_tuple() == pytest.approx((2, 2, expected_theta))
        
    def test_move_5p(self):
        self.create_logic_provider()
        
        file_path = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.abspath(os.path.join(file_path, 'test_data.yaml'))
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            start_arr_content = data['particles_5_start']
            end_arr_content = data['particles_5_end']
            
        start_arr = np.array(start_arr_content)
        end_arr = np.array(end_arr_content)
        
        particles = self.array_to_particles(start_arr)

        N_start = (2, 1, math.pi / 4)
        N_end = (2, 2, math.pi / 4)
        d_x, d_y, d_t = N_end[0] - N_start[0], N_end[1] - N_start[1], N_end[2] - N_start[2]
        new_particles = self.logic_provider.update_particles_with_odom((d_x, d_y, d_t), N_start, particles)
        out_array = self.particles_to_array(new_particles)
        
        assert out_array == pytest.approx(end_arr)
        
    def test_move_many(self):
        self.create_logic_provider()
        
        file_path = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.abspath(os.path.join(file_path, 'test_data.yaml'))
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            start_arr_content = data['particles_many_start']
            end_arr_content = data['particles_many_end']        
        
        start_arr = np.array(start_arr_content)
        end_arr = np.array(end_arr_content)
        particles = self.array_to_particles(start_arr)

        N_start = (2, 1, math.pi / 6)
        N_end = (2, 2, math.pi / 4)
        d_x, d_y, d_t = N_end[0] - N_start[0], N_end[1] - N_start[1], N_end[2] - N_start[2]
        new_particles = self.logic_provider.update_particles_with_odom((d_x, d_y, d_t), N_start, particles)
        out_array = self.particles_to_array(new_particles)
        
        assert out_array == pytest.approx(end_arr)
        

class TestInitializeParticleCloud(TestBase):
    def test_number(self):
        self.create_logic_provider()
        
        particles = self.logic_provider.initialize_particle_cloud(0)
        assert len(particles) == self.n_particles
        
    def test_avg_uniform(self, plt):
        self.create_logic_provider(n_particles=1000)
        particles = self.logic_provider.initialize_particle_cloud(0)
        
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        center = ((bounds[0][1]+bounds[0][0])/2, (bounds[1][1]+bounds[1][0])/2, math.pi)
        x_vals = [p.x for p in particles]
        y_vals = [p.y for p in particles]
        theta_vals = [p.theta for p in particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        avg_theta = sum(theta_vals) / len(theta_vals)
        error_x = avg_x-center[0]
        error_y = avg_y-center[1]
        error_theta = avg_theta-center[2]
        
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(center[0], center[1], c="black", label="center")
        plt.title("uniform dist")
        plt.legend()
        
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1
        assert abs(error_theta) <= 0.15
        
    def test_avg_with_guess(self, plt):
        self.create_logic_provider(n_particles=2000)
    
        guess = (1, -2, 0)
        
        particles = self.logic_provider.initialize_particle_cloud(0, guess)
        
        x_vals = [p.x for p in particles]
        y_vals = [p.y for p in particles]
        theta_vals = [p.theta for p in particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        error_x = avg_x-guess[0]
        error_y = avg_y-guess[1]
        
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(guess[0], guess[1], c="black", label="guess")
        plt.title("initialized with a guess")
        plt.legend()
        
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1


class TestResampleParticles(TestBase):
    def test_number(self):
        self.create_logic_provider()
        
        particles = self.logic_provider.initialize_particle_cloud(0)
        new_particles = self.logic_provider.resample_particles(particles)
        assert len(new_particles) == self.n_particles
        
    def test_1_weighted(self, plt):
        self.create_logic_provider(n_particles_uniform=0)
        #don't include uniform particles because they will drag the center away from the guess point
        
        particles = [Particle(w=0) for _ in range(10000)]
        pos = (1, -2)
        particles[0].w = 1
        particles[0].x = pos[0]
        particles[0].y = pos[1]
        
        new_particles = self.logic_provider.resample_particles(particles)
        x_vals = [p.x for p in new_particles]
        y_vals = [p.y for p in new_particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        error_x = avg_x-pos[0]
        error_y = avg_y-pos[1]
        
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(pos[0], pos[1], c="black", label="set point")
        plt.title("sampled on 1 weighted point")
        plt.legend()
        
        assert abs(error_x) <= 0.05
        assert abs(error_y) <= 0.05
        
    def test_3_weighted(self, plt):
        self.create_logic_provider()
        #the uniform particles are ok here because our target center is in the center of the bounds
        
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        center = ((bounds[0][1]+bounds[0][0])/2, (bounds[1][1]+bounds[1][0])/2, math.pi)
        particles = [Particle(w=0) for _ in range(10000)]
        d = 0.5
        pos = ((center[0]+d, center[1]), 
               (center[0]+d*math.cos(2*math.pi/3), center[1]+d*math.sin(2*math.pi/3)), 
               (center[0]+d*math.cos(4*math.pi/3), center[1]+d*math.sin(4*math.pi/3)))
        particles[0].w = 1/3
        particles[1].w = 1/3
        particles[2].w = 1/3
        particles[0].x = pos[0][0]
        particles[1].x = pos[1][0]
        particles[2].x = pos[2][0]
        particles[0].y = pos[0][1]
        particles[1].y = pos[1][1]
        particles[2].y = pos[2][1]

        
        new_particles = self.logic_provider.resample_particles(particles)
        x_vals = [p.x for p in new_particles]
        y_vals = [p.y for p in new_particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        error_x = avg_x-center[0]
        error_y = avg_y-center[1]
        
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(center[0], center[1], c="purple", label="center")
        plt.scatter([pos[0][0], pos[1][0], pos[2][0]], [pos[0][1], pos[1][1], pos[2][1]], c="black", label="weighted points")
        plt.title("sampled on 3 weighted points")
        plt.legend()
        
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1
        
    def test_different_weights(self, plt):
        self.create_logic_provider()
        particles = [Particle(w=0) for _ in range(10000)]
        
        def set_weighted_points(n_points, dist, center, weights_list):
            poses = []
            angle_part = 2*math.pi/n_points
            for i in range(n_points):
                poses.append((center[0]+math.cos(angle_part*i)*dist, center[1]+math.sin(angle_part*i)*dist))
                p = particles[i]
                p.w = weights_list[i]
                p.x, p.y = poses[i]
            return poses
        
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        center = ((bounds[0][1]+bounds[0][0])/2, (bounds[1][1]+bounds[1][0])/2, math.pi)
        n_points = 5
        dist = 0.5
        weights_list = [0.05, 0.1, 0.5, 0.25, 0.1]
        weighted_poses = set_weighted_points(n_points, dist, center, weights_list)
        
        new_particles = self.logic_provider.resample_particles(particles)
        x_vals = [p.x for p in new_particles]
        y_vals = [p.y for p in new_particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        

        weighted_center_x = sum([weighted_poses[i][0]*weights_list[i] for i in range(n_points)])
        weighted_center_y = sum([weighted_poses[i][1]*weights_list[i] for i in range(n_points)])
        
        error_x = avg_x-weighted_center_x
        error_y = avg_y-weighted_center_y
        
        
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(weighted_center_x, weighted_center_y, c="purple", label="weighted_center")
        plt.scatter([p[0] for p in weighted_poses], [p[1] for p in weighted_poses], c="black", label="weighted points")
        plt.title("sampled on 5 differently weighted points")
        plt.legend()
        plt.axis('equal')
        
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1
            

        
        
        

class TestUpdateParticlesWithLaser(TestBase):
    def initialize_data(self):
        """ Initialize enough variables to test with. """
        super().create_logic_provider()
        self.particle_cloud = self.logic_provider.initialize_particle_cloud(0)
        self.theta = [i for i in range(361)]

        # Randomly choose the distance to the closest object to the robot
        # from 0 to 3 meters.
        min_dist = random.random() * 3

        # Make a random laser scan with distances
        # from 0-3 meters to 5 meters.
        self.r = [random.random() * (5 - min_dist) + min_dist for _ in range(360)]
        self.r.append(self.r[0])

        # Make sure the scan is in the right range.
        assert 0 <= min_dist <= 3
        assert all(min_dist <= r <= 5 for r in self.r)

        # Update the particle weights.
        (self.particle_cloud, _) = self.logic_provider.update_particles_with_laser(
            self.r, self.theta, self.particle_cloud
        )

    @pytest.mark.repeat(10)
    def test_positive(self):
        """ Test that all the particle weights are positive. """
        self.initialize_data()
        assert all(p.w >= 0 for p in self.particle_cloud)

    @pytest.mark.repeat(10)
    def test_float(self):
        """ Test that all the particle weights are floats. """
        self.initialize_data()
        assert all(isinstance(p.w, float) for p in self.particle_cloud)

    @pytest.mark.repeat(10)
    def test_not_nan(self):
        """ Test that all the particle weights are not nan. """
        self.initialize_data()
        assert all(not math.isnan(p.w) for p in self.particle_cloud)

class TestNormalize(TestBase):
    def initialize_data(self):
        """ Initialize enough variables to test with. """
        super().create_logic_provider()
        self.particle_cloud = self.logic_provider.initialize_particle_cloud(0)

    @pytest.mark.repeat(10)
    def test_sum_is_one(self):
        self.initialize_data()

        # Randomize the weights of the particles.
        for p in self.particle_cloud:
            p.w = random.random() * 100

        # Normalize their weights.
        self.particle_cloud = self.logic_provider.normalize(self.particle_cloud)

        assert sum(p.w for p in self.particle_cloud) == pytest.approx(1)

class TestUpdateRobotPose(TestBase):
    def initialize_data(self):
        """ Initialize enough variables to test with. """
        super().create_logic_provider()
        self.particle_cloud = self.logic_provider.initialize_particle_cloud(0)
        self.theta = [i for i in range(361)]

        # Randomly choose the distance to the closest object to the robot
        # from 0 to 3 meters.
        min_dist = random.random() * 3

        # Make a random laser scan with distances
        # from 0-3 meters to 5 meters.
        self.r = [random.random() * (5 - min_dist) + min_dist for _ in range(360)]
        self.r.append(self.r[0])

        # Make sure the scan is in the right range.
        assert 0 <= min_dist <= 3
        assert all(min_dist <= r <= 5 for r in self.r)
        self.particle_cloud = self.logic_provider.normalize(self.particle_cloud)

        # Update the particle weights.
        (self.particle_cloud, _) = self.logic_provider.update_particles_with_laser(
            self.r, self.theta, self.particle_cloud
        )

    @pytest.mark.repeat(10)
    def test_highest_weight(self):
        """
        Assert that the robot pose is that of the highest weighted particle.
        """
        self.initialize_data()

        # Calculate the robot's pose
        robot_pose = self.logic_provider.update_robot_pose(self.particle_cloud)

        # Make a list of the highest weighted particles,
        # one if there is a highest more if there is a tie.
        weight_max = max(p.w for p in self.particle_cloud)
        #print(f"{weight_max = }")
        best_list = []
        for p in self.particle_cloud:
            if p.w == weight_max:
                best_list.append(p)

        assert len(best_list) > 0
        #print(f"{len(best_list) = }")

        # For each particle in the list:
        for p in best_list:
            # Check if it has the same position and angle as the robot.
            same_x = abs(robot_pose.position.x - p.x) < 1e-6
            same_y = abs(robot_pose.position.y - p.y) < 1e-6
            (roll, pitch, theta) = euler_from_quaternion(
                robot_pose.orientation.x,
                robot_pose.orientation.y,
                robot_pose.orientation.z,
                robot_pose.orientation.w
            )
            same_theta = abs((theta % (2*math.pi)) - (p.theta % (2*math.pi))) < 1e-6

            #print(f"{roll = }")
            #print(f"{pitch = }")
            #print(f"{robot_pose.position.x = }")
            #print(f"{p.x = }")
            #print(f"{robot_pose.position.y = }")
            #print(f"{p.y = }")
            #print(f"{theta = }")
            #print(f"{p.theta = }")

            # If all are the same, pass the assertion
            if same_x and same_y and same_theta:
                #print("pass\n")
                return
            
            #print("fail\n")

        # If a best particle that matches the robot pose hasn't been found:
        # Fail the assertion.
        assert False
