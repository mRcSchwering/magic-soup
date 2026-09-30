import json
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"


def load_test_file(filename: str) -> bytes:
    path = TEST_DATA_DIR / filename
    return path.read_bytes()


def load_test_json(filename: str) -> dict:
    return json.loads(load_test_file(filename).decode("utf-8"))
