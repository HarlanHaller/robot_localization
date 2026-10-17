import pytest
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
