import pytest
import random
import math
from robot_localization.robot_localization.logic_provider import LogicProvider
from robot_localization.tests.test_support_classes.support_occupancy_field import OccupancyField
from robot_localization.tests.test_support_classes.support_parameter import Parameter

class TestBase:
    def __init__(self, n_particles = 300, n_particles_uniform = 50):
        self.occupancy_field = OccupancyField("./mapdata/gauntlet.pkl")
        self.n_particles = n_particles
        self.n_particles_uniform_param = Parameter(n_particles_uniform)
        self.logic_provider = LogicProvider(
            self.occupancy_field, self.n_particles, self.n_particles_uniform_param)


class TestResample(TestBase):
    def __init__(self):
        super().__init__()

    def test_part1(self):
        assert 1.00000001 == pytest.approx(1)

    def test_part2(self):
        assert 0+1 == 2


class TestInitialize(TestBase):
    def __init__(self):
        super().__init__()

    def test_part1(self):
        assert 1==1

class TestUpdateParticlesWithLaser(TestBase):
    def __init__(self):
        """ Initialize enough variables to test with """
        super().__init__()
        self.particle_cloud = LogicProvider.initialize_particle_cloud(self, 0)
        self.theta = [i for i in range(361)]

    def take_random_scan(self):
        """
        Make a random laser scan.
        """
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

    def retest(self, assertion):
        """
        Make 10 random scans and for each one assert that assertion is true.

        Args:
            assertion: A function that takes in self.particle_cloud
            representing the assertion to assert.
        """
        for _ in range(10):
            self.take_random_scan()
            (self.particle_cloud, _) = LogicProvider.update_particles_with_laser(
                self.r, self.theta, self.particle_cloud)
            assert assertion(self.particle_cloud)

    def test_positive(self):
        """ Test that all the particle weights are positive. """
        def all_positive(particle_cloud):
            return all(p.w >= 0 for p in particle_cloud)

        self.retest(all_positive)

    def test_float(self):
        """ Test that all the particle weights are floats. """
        def all_floats(particle_cloud):
            return all(isinstance(p.w, float) for p in particle_cloud)

        self.retest(all_floats)

    def test_sum_to_one(self):
        """ Test that the sum of all the particle weights is 1. """
        def sum_is_one(particle_cloud):
            return sum(p.w for p in particle_cloud) == pytest.approx(1)

        self.retest(sum_is_one)

    def test_not_nan(self):
        """ Test that all the particle weights are not nan. """
        def none_are_nan(particle_cloud):
            return all(not math.isnan(p.w) for p in particle_cloud)

        self.retest(none_are_nan)
