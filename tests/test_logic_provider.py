import math
import os
import random
from typing import List
import pytest
import numpy as np
import yaml
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

class TestUpdateParticlesWithLaser(TestBase):
    def initialize_data(self):
        """ Initialize enough variables to test with """
        super().create_logic_provider()
        self.particle_cloud = LogicProvider.initialize_particle_cloud(self, 0)
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
