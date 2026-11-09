import numpy as np
import pytest

from crashdamage import dataset


@pytest.fixture(scope="session")
def small_data():
    return dataset.generate(400, seed=5)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
