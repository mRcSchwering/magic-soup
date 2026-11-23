import random

import pytest
import torch

from tests.config import DEBUG, RNG_SEED

if DEBUG and torch.cuda.is_available():
    torch.cuda.set_sync_debug_mode("warn")


@pytest.fixture
def rng() -> random.Random:
    return random.Random(RNG_SEED)
