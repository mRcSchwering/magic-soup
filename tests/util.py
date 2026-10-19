import json
from pathlib import Path
from typing import Any

import torch

from tests.config import DEVICE, FTYPE, IDXTYPE, ITYPE

TEST_DATA_DIR = Path(__file__).parent / "data"


def seed(s: int = 42) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)


def load_test_file(filename: str) -> bytes:
    path = TEST_DATA_DIR / filename
    return path.read_bytes()


def is_moore_nbd(a: tuple[int, int], b: tuple[int, int], n: int) -> bool:
    """
    Check if two points are neighbors in a Moore neighborhood on a period n x n grid.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    # Shortest wrapped distance on each axis
    dx = min(dx, n - dx)
    dy = min(dy, n - dy)

    return max(dx, dy) <= 1 and (dx, dy) != (0, 0)


def load_test_json(filename: str) -> dict:
    return json.loads(load_test_file(filename).decode("utf-8"))


def idxtensor(d: Any) -> torch.Tensor:
    return torch.tensor(d, device=DEVICE, dtype=IDXTYPE)


def itensor(d: Any) -> torch.Tensor:
    return torch.tensor(d, device=DEVICE, dtype=ITYPE)


def ftensor(d: Any) -> torch.Tensor:
    return torch.tensor(d, device=DEVICE, dtype=FTYPE)


def idxzeros(*args) -> torch.Tensor:
    return torch.zeros(*args, device=DEVICE, dtype=IDXTYPE)


def izeros(*args) -> torch.Tensor:
    return torch.zeros(*args, device=DEVICE, dtype=ITYPE)


def fzeros(*args) -> torch.Tensor:
    return torch.zeros(*args, device=DEVICE, dtype=FTYPE)


def idxfull(*args, value: int) -> torch.Tensor:
    return torch.full(args, fill_value=value, device=DEVICE, dtype=IDXTYPE)


def ffull(*args, value: float) -> torch.Tensor:
    return torch.full(args, fill_value=value, device=DEVICE, dtype=FTYPE)


def frand(*args) -> torch.Tensor:
    return torch.rand(*args, device=DEVICE, dtype=FTYPE)


def irand(low: int, high: int, *args) -> torch.Tensor:
    return torch.randint(low, high, args, device=DEVICE, dtype=ITYPE)


def idxrandperm(n: int) -> torch.Tensor:
    return torch.randperm(n, device=DEVICE, dtype=IDXTYPE)


def idxarange(n: int) -> torch.Tensor:
    return torch.arange(n, device=DEVICE, dtype=IDXTYPE)
