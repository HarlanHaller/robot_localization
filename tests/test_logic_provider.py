""" Test logic_provider.py """
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
    """
    This serves as a base for all tests which simply
    defines a method to initialize the logic provider.
    """
    def create_logic_provider(self, n_particles = 0, n_particles_uniform=None):
        """
        Initialize a LogicProvider.
        """
        # Set the map.
        self.occupancy_field = OccupancyField("mapdata/gauntlet.pkl")

        # Set the constants from a YAML file.
        file_path = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.abspath(os.path.join(file_path, 'config.yaml'))
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            if n_particles == 0:
                n_particles = data["n_particles"]
            if n_particles_uniform == None:
                n_particles_uniform = math.floor(n_particles*data["percent_particles_uniform"])
        self.n_particles = n_particles
        self.n_particles_uniform_param = Parameter(n_particles_uniform)
        self.logic_provider = LogicProvider(
            self.occupancy_field, yaml_path) # pyright: ignore[reportArgumentType]
        self.logic_provider.n_particles = n_particles
        self.logic_provider.n_particles_uniform = n_particles_uniform


class TestUpdateWithOdom(TestBase):
    """
    This class groups all the tests for the update_particles_with_odom method.
    """
    def array_to_particles(self, arr) -> List[Particle]:
        """
        This is a helper method to convert a array
        of column vectors [x; y; theta] into a list of particles.

        Args:
            arr (ND array): The array of particles to provide data

        Returns:
            List[Particle]: A list of particles
        """
        particles: List[Particle] = []
        for i in range(arr.shape[1]):
            p = Particle(x = arr[0, i], y=arr[1,i], theta=arr[2,i])
            particles.append(p)
        return particles

    def particles_to_array(self, plist):
        """
        This helper function converts a list of particles
        into a np array of column vectors [x; y; theta]

        Args:
            plist (List[particles]): the list of particles to convert

        Returns:
            ND array: a array with the data from the list of particles
        """
        arr = np.array([[p.x, p.y, p.theta] for p in plist]).T
        return arr

    def test_move_1p_1(self):
        """
        This test tests moving a single point
        """
        self.create_logic_provider()

        # Initialize the test particle
        particles = [Particle(x = 1, y = 2, theta = 3 * math.pi / 2)]
        # Calculate the delta
        n_start = (2, 1, math.pi / 2)
        n_end = (2, 2, math.pi / 6)
        d_x, d_y, d_t = n_end[0] - n_start[0], n_end[1] - n_start[1], n_end[2] - n_start[2]
        # Update the particle
        new_particles = self.logic_provider.update_particles_with_odom(
            (d_x, d_y, d_t), n_start, particles)

        expected_theta = 3 * math.pi / 2 + d_t
        # compare to expected value, caluclated with matlab
        assert new_particles[0].as_tuple() == pytest.approx((1, 1, expected_theta))

    def test_move_1p_2(self):
        """This test tests moving a single point"""
        self.create_logic_provider()

        particles = [Particle(x = 1, y = 2, theta = 0)]
        n_start = (2, 1, math.pi / 2)
        n_end = (2, 2, math.pi / 6)
        d_x, d_y, d_t = n_end[0] - n_start[0], n_end[1] - n_start[1], n_end[2] - n_start[2]
        new_particles = self.logic_provider.update_particles_with_odom(
            (d_x, d_y, d_t), n_start, particles)

        expected_theta = 0 + d_t

        assert new_particles[0].as_tuple() == pytest.approx((2, 2, expected_theta))

    def test_move_5p(self):
        """This method tests moving 5 points at once. Uses data calculated with matlab"""
        self.create_logic_provider()

        #Load the starting positon and expeted end positions from a yaml
        file_path = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.abspath(os.path.join(file_path, 'test_data.yaml'))

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            start_arr_content = data['particles_5_start']
            end_arr_content = data['particles_5_end']

        start_arr = np.array(start_arr_content)
        end_arr = np.array(end_arr_content)

        # Convert the matrix of data into a list of particles
        particles = self.array_to_particles(start_arr)

        # Calculates the delta values
        N_start = (2, 1, math.pi / 4)
        N_end = (2, 2, math.pi / 4)
        d_x, d_y, d_t = N_end[0] - N_start[0], N_end[1] - N_start[1], N_end[2] - N_start[2]
        new_particles = self.logic_provider.update_particles_with_odom(
            (d_x, d_y, d_t), N_start, particles)

        #converts the particle list back to an array for comparison
        out_array = self.particles_to_array(new_particles)

        assert out_array == pytest.approx(end_arr)

    def test_move_many(self):
        """
        This method tests moving many points at once. Uses data calculated with matlab.
        It is the same as the test with 5 particles accept for a slightly different
        start position and more particles used.
        """
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
        new_particles = self.logic_provider.update_particles_with_odom(
            (d_x, d_y, d_t), N_start, particles)
        out_array = self.particles_to_array(new_particles)

        assert out_array == pytest.approx(end_arr)


class TestInitializeParticleCloud(TestBase):
    """
    This class groups all the tests for the initialize_particle_cloud function
    """

    def angle_normalize(self, z):
        """ convenience function to map an angle to the range [-pi,pi]
        Copied form helper_function, avoid needing create a TFHelper object"""
        return math.atan2(math.sin(z), math.cos(z))

    def angle_diff(self, a, b):
        """ Calculates the difference between angle a and angle b (both should
            be in radians) the difference is always based on the closest
            rotation from angle a to angle b.
            examples:
                angle_diff(.1,.2) -> -.1
                angle_diff(.1, 2*math.pi - .1) -> .2
                angle_diff(.1, .2+2*math.pi) -> -.1
            Copied form helper_function, avoid needing create a TFHelper object
        """
        a = self.angle_normalize(a)
        b = self.angle_normalize(b)
        d1 = a-b
        d2 = 2*math.pi - math.fabs(d1)
        if d1 > 0:
            d2 *= -1.0
        if math.fabs(d1) < math.fabs(d2):
            return d1
        else:
            return d2

    def test_number(self):
        """Tests if the particle filter initializes with the right number of particles
        """
        self.create_logic_provider()

        particles = self.logic_provider.initialize_particle_cloud(0)
        assert len(particles) == self.n_particles

    def test_avg_uniform(self, plt):
        """
        Test if the particles generated uniformly have a "center of mass" in the middle of the map.
        Additonaly plots the points to visualy confirm that particles are distrobuted as epxected. 

        Args:
            plt: the plot fixture provided by pytest-plt.
            When pytest --plots is run it will generate plots
        """
        self.create_logic_provider(n_particles=1000)
        particles = self.logic_provider.initialize_particle_cloud(0)

        # Calculates the center of the map
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        center = ((bounds[0][1]+bounds[0][0])/2, (bounds[1][1]+bounds[1][0])/2, math.pi)

        # get the x, y and theta values of generated particles
        x_vals = [p.x for p in particles]
        y_vals = [p.y for p in particles]
        theta_vals = [p.theta for p in particles]

        # Compute the averages
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        avg_theta = sum(theta_vals) / len(theta_vals)

        # Find error to expected values
        error_x = avg_x-center[0]
        error_y = avg_y-center[1]
        error_theta = avg_theta-center[2]

        # Make a plot of all the particles
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(center[0], center[1], c="black", label="center")
        plt.title("uniform dist")
        plt.legend()

        # Check that all error values are sufficently small
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1
        assert abs(error_theta) <= 0.15

    def test_avg_with_guess(self, plt):
        """Test if the particles generated with a guess has a "center of mass"
        centered around the provided guess. Additonaly plots the points to visualy 
        confirm that particles are distrobuted as epxected. 
        
        Args:
            plt: the plot fixture provided by pytest-plt.
            When pytest --plots is run it will generate plots
        """
        self.create_logic_provider(n_particles=2000)

        guess = (1, -2, 0)

        particles = self.logic_provider.initialize_particle_cloud(0, guess)

        # Get the x, y, theta values of the generated
        x_vals = [p.x for p in particles]
        y_vals = [p.y for p in particles]
        # taking the average of theta values causes issues, instead coputer distance directly
        theta_diff = [self.angle_diff(p.theta, guess[2]) for p in particles]

        #find the averages
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        avg_theta_diff = sum(theta_diff) / len(theta_diff)

        #compute the errors with the guess
        error_x = avg_x-guess[0]
        error_y = avg_y-guess[1]

        # Generate the plot of particle positions
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(guess[0], guess[1], c="black", label="guess")
        plt.title("initialized with a guess")
        plt.legend()

        # Assert that the error is low
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1
        assert abs(avg_theta_diff) <= 0.01


class TestResampleParticles(TestBase):
    """This class groups all of the tests for the resample_particles method
    """
    def test_number(self):
        """This function ensures that after resampling the number of particles remains correct
        """
        self.create_logic_provider()

        particles = self.logic_provider.initialize_particle_cloud(0)
        new_particles = self.logic_provider.resample_particles(particles)
        assert len(new_particles) == self.n_particles

    def test_1_weighted(self, plt):
        """Test if the when resampling based one particle the resulting particles 
        are centered on that particle

        Args:
            plt: the plot fixture provided by pytest-plt.
            When pytest --plots is run it will generate plots
        """
        self.create_logic_provider(n_particles_uniform=0)
        #don't include uniform particles because they will drag the center away from the guess point

        #create a list of 10k particles with wieght = 0
        particles = [Particle(w=0) for _ in range(10000)]
        pos = (1, -2)

        # initialize one particle with w=1 and a chosen position
        particles[0].w = 1
        particles[0].x = pos[0]
        particles[0].y = pos[1]

        # Resample the particles
        new_particles = self.logic_provider.resample_particles(particles)

        # get the x and y values from all new particles
        x_vals = [p.x for p in new_particles]
        y_vals = [p.y for p in new_particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)

        #Calcualte the error values
        error_x = avg_x-pos[0]
        error_y = avg_y-pos[1]

        #Generate the plot
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(pos[0], pos[1], c="black", label="set point")
        plt.title("sampled on 1 weighted point")
        plt.legend()

        #assert that the errors are low
        assert abs(error_x) <= 0.05
        assert abs(error_y) <= 0.05

    def test_3_weighted(self, plt):
        """Test if when we resample with 3 equally weighted particles the resulting particles 
        are centered at the middle of thoes 3 particles

        Args:
            plt: the plot fixture provided by pytest-plt.
            When pytest --plots is run it will generate plots
        """
        self.create_logic_provider()
        #the uniform particles are ok here because our target center is in the center of the bounds

        #calcualte the center point
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        center = ((bounds[0][1]+bounds[0][0])/2, (bounds[1][1]+bounds[1][0])/2, math.pi)

        # Initialize particles
        particles = [Particle(w=0) for _ in range(10000)]

        #calcualte and set the particles to position of equal lateral triangle
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

        # Resample particles and calculate the error values.
        new_particles = self.logic_provider.resample_particles(particles)
        x_vals = [p.x for p in new_particles]
        y_vals = [p.y for p in new_particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        error_x = avg_x-center[0]
        error_y = avg_y-center[1]

        # Make the plot
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(center[0], center[1], c="purple", label="center")
        plt.scatter([pos[0][0], pos[1][0], pos[2][0]], [pos[0][1], pos[1][1], pos[2][1]],
                    c="black", label="weighted points")
        plt.title("sampled on 3 weighted points")
        plt.legend()

        # assert that the error values are low
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1

    def test_different_weights(self, plt):
        """
        Test that if we resample with 5 particle with different weights that
        The resulting particles are center around the weighted average of the initial particles

        Args:
            plt: the plot fixture provided by pytest-plt.
            When pytest --plots is run it will generate plots
        """
        self.create_logic_provider()
        # Create a list of particles
        particles = [Particle(w=0) for _ in range(10000)]

        def set_weighted_points(n_points, dist, center, weights_list):
            """returns a set of poses for a n-agon given a center point
            and a list of weights for the particles, and the distance form 
            the center for each particle.

            Args:
                n_points (int): the number of points (the n in the n-agon)
                dist (float): the distance from the center point (radius of circumcircle)
                center (Tuple[float]): A tuple (x, y) for the center point of the shape
                weights_list (List[float]): a list of floats to be set as the weights for 
                each particle

            Returns:
                List[Tuple]: A list of tuples for each vertex on the n-agon
            """
            poses = []
            angle_part = 2*math.pi/n_points
            for i in range(n_points):
                poses.append((
                    center[0]+math.cos(angle_part*i)*dist,
                     center[1]+math.sin(angle_part*i)*dist
                ))
                p = particles[i]
                p.w = weights_list[i]
                p.x, p.y = poses[i]
            return poses

        # Find the center
        bounds = self.occupancy_field.get_obstacle_bounding_box()
        center = ((bounds[0][1]+bounds[0][0])/2, (bounds[1][1]+bounds[1][0])/2, math.pi)

        # Define paramaters for the distrobution of initial points
        n_points = 5
        dist = 0.5
        weights_list = [0.05, 0.1, 0.5, 0.25, 0.1]
        # Calcuate positions for a pentagon of particles of different weights
        weighted_poses = set_weighted_points(n_points, dist, center, weights_list)

        # resample the particles ang find average position
        new_particles = self.logic_provider.resample_particles(particles)
        x_vals = [p.x for p in new_particles]
        y_vals = [p.y for p in new_particles]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)

        # find the expected weighted center point and the error to the expected weighted center
        weighted_center_x = sum([weighted_poses[i][0]*weights_list[i] for i in range(n_points)])
        weighted_center_y = sum([weighted_poses[i][1]*weights_list[i] for i in range(n_points)])

        error_x = avg_x-weighted_center_x
        error_y = avg_y-weighted_center_y

        # make the plot
        plt.scatter(x_vals, y_vals, label = "points")
        plt.scatter(avg_x, avg_y, c="red", label="averge")
        plt.scatter(weighted_center_x, weighted_center_y, c="purple", label="weighted_center")
        plt.scatter([p[0] for p in weighted_poses], [p[1] for p in weighted_poses],
                    c="black", label="weighted points")
        plt.title("sampled on 5 differently weighted points")
        plt.legend()
        plt.axis('equal')

        # assert that the error is low
        assert abs(error_x) <= 0.1
        assert abs(error_y) <= 0.1


class TestUpdateParticlesWithLaser(TestBase):
    """ Contains tests for LogicProvider.update_particles_with_laser """
    def initialize_data(self):
        """
        Initialize a particle cloud and laser scan
        and calculate the particle weights.
        """
        super().create_logic_provider()
        # Make a particle cloud with uniform random poses.
        self.particle_cloud = self.logic_provider.initialize_particle_cloud(0)

        deg_to_rad = math.pi / 180
        self.theta = [i*deg_to_rad for i in range(361)]

        # Randomly choose the distance to the closest object to the robot
        # from 0 to 1 meters.
        min_dist = random.random()

        # Make a random laser scan with distances
        # from min_dist meters to 3 meters.
        self.r = [random.random() * (3 - min_dist) + min_dist for _ in range(360)]
        self.r.append(self.r[0])

        # For speed, only use some of the scan.
        reduced_r = self.r[0:360:10]
        reduced_theta = self.theta[0:360:10]

        # Make sure the scan is in the right range.
        assert 0 <= min_dist <= 1
        assert all(min_dist <= r <= 5 for r in self.r)

        # Update the particle weights.
        self.particle_cloud = self.logic_provider.update_particles_with_laser(
            reduced_r, reduced_theta, self.particle_cloud
        )

    def test_boundary_enforcement(self):
        """
        Assert that particles that are on the map have
        a weight of zero and that those on the map aren't.
        """
        self.create_logic_provider()

        # Initialize an arbitrary laser scan so the weights can be updated.
        r = [0.5]
        theta = [0]

        # Make particles that are out of bounds and one that is in bounds.
        oob_x_min = Particle(x=-10, y=0, theta=0)
        oob_x_max = Particle(x=10, y=0, theta=0)
        oob_y_min = Particle(x=0, y=-10, theta=0)
        oob_y_max = Particle(x=0, y=10, theta=0)
        ib = Particle(x=0, y=0, theta=0)
        p_list = [oob_x_min, oob_x_max, oob_y_min, oob_y_max, ib]

        # Calculate the particles' weights.
        new_particles = self.logic_provider.update_particles_with_laser(r, theta, p_list)

        # Assert that particles that are on the map
        # have a weight of zero and that those on the map aren't.
        assert new_particles[0].w == 0
        assert new_particles[1].w == 0
        assert new_particles[2].w == 0
        assert new_particles[3].w == 0
        assert new_particles[4].w >= 0

    @pytest.mark.repeat(3)
    def test_positive(self):
        """ Test that all the particle weights are positive. """
        self.initialize_data()
        assert all(p.w >= 0 for p in self.particle_cloud)

    @pytest.mark.repeat(3)
    def test_float(self):
        """ Test that all the particle weights are floats. """
        self.initialize_data()
        assert all(isinstance(p.w, float) for p in self.particle_cloud)

    @pytest.mark.repeat(3)
    def test_not_nan(self):
        """ Test that all the particle weights are not nan. """
        self.initialize_data()
        assert all(not math.isnan(p.w) for p in self.particle_cloud)

    def test_weight_v_index(self, plt):
        """ Graph the distribution of random particle weights """
        # Take weights from n random samples:
        n = 10
        weights = []
        for _ in range(n):
            self.initialize_data()
            weights += [p.w for p in self.particle_cloud]

        # Sort the weights and graph them by index.
        weights.sort()
        plt.plot(weights)
        plt.title(f"Weights of particles from {n} randomly generated scans")
        plt.xlabel("index (sorted)")
        plt.ylabel("weight (not normalized)")

    def test_histogram(self, plt):
        """ Graph the distribution of random particle weights """
        # Take weights from n random samples:
        n = 10
        weights = []
        for _ in range(n):
            self.initialize_data()
            weights += [p.w for p in self.particle_cloud]

        # Make a histogram of the weights.
        plt.hist(weights, 25)
        plt.title(f"Probability of each weight from {n} randomly generated scans")
        plt.xlabel("weights (not normalized)")
        plt.ylabel(f"number found (from {n} particle clouds at different robot poses)")

class TestNormalize(TestBase):
    """ Contains tests for LogicProvider.normalize """
    def initialize_data(self):
        """ Initialize a particle cloud. """
        super().create_logic_provider()
        self.particle_cloud = self.logic_provider.initialize_particle_cloud(0)

    @pytest.mark.repeat(3)
    def test_sum_is_one(self):
        """
        Assert that the weights of the particle cloud
        sum to one after being normalized.
        """
        self.initialize_data()

        # Randomize the weights of the particles.
        for p in self.particle_cloud:
            p.w = random.random() * 100

        # Normalize their weights.
        self.particle_cloud = self.logic_provider.normalize(self.particle_cloud)

        # Assert that the weights sum to one.
        assert sum(p.w for p in self.particle_cloud) == pytest.approx(1)

class TestUpdateRobotPose(TestBase):
    """ Contains tests for LogicProvider.update_robot_pose """
    def initialize_data(self):
        """
        Initialize a particle cloud and laser scan
        and calculate the particle weights.
        """
        super().create_logic_provider()
        # Make a particle cloud with uniform random poses.
        self.particle_cloud = self.logic_provider.initialize_particle_cloud(0)

        deg_to_rad = math.pi / 180
        self.theta = [i*deg_to_rad for i in range(361)]

        # Randomly choose the distance to the closest object to the robot
        # from 0 to 1 meters.
        min_dist = random.random()

        # Make a random laser scan with distances
        # from min_dist meters to 3 meters.
        self.r = [random.random() * (3 - min_dist) + min_dist for _ in range(360)]
        self.r.append(self.r[0])

        # For speed, only use some of the scan.
        reduced_r = self.r[0:360:10]
        reduced_theta = self.theta[0:360:10]

        # Make sure the scan is in the right range.
        assert 0 <= min_dist <= 1
        assert all(min_dist <= r <= 5 for r in self.r)

        # Update the particle weights.
        self.particle_cloud = self.logic_provider.update_particles_with_laser(
            reduced_r, reduced_theta, self.particle_cloud
        )

    @pytest.mark.repeat(3)
    def test_highest_weight(self):
        """
        Assert that the robot pose is that of the highest weighted particle.
        """
        self.initialize_data()

        # Calculate the robot's pose
        robot_pose = self.logic_provider.update_robot_pose(self.particle_cloud)

        # Assert that a best particle was found.
        assert robot_pose is not None

        # Make a list of the highest weighted particles,
        # one if there is a highest more if there is a tie.
        weight_max = max(p.w for p in self.particle_cloud)
        best_list = []
        for p in self.particle_cloud:
            if p.w == weight_max:
                best_list.append(p)

        # Assert that a best particle was found.
        assert len(best_list) > 0

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

            # If all are the same, pass the assertion
            if same_x and same_y and same_theta:
                return

        # If a best particle that matches the robot pose hasn't been found:
        # Fail the assertion.
        assert False
