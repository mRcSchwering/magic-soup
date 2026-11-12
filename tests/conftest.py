import random

import pytest

from tests.config import RNG_SEED


@pytest.fixture
def rng() -> random.Random:
    return random.Random(RNG_SEED)
