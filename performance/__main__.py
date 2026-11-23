import argparse
from collections.abc import Callable

from performance import check, simulation

_cmds: dict[str, Callable] = {
    "check": check.main,
    "simulation": simulation.main,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    check_parser = subparsers.add_parser("check", help=check.__doc__)
    check_parser.add_argument("--parts", default=check.default_parts, nargs="*", action="store")
    check_parser.add_argument("--n-cells", default=10_000, type=int)
    check_parser.add_argument("--genome-size", default=1_000, type=int)
    check_parser.add_argument("--device", default="cpu", type=str)

    sim_parser = subparsers.add_parser("simulation", help=simulation.__doc__)
    sim_parser.add_argument("--map-size", default=128, type=int)
    sim_parser.add_argument("--n-steps", default=200, type=int)
    sim_parser.add_argument("--init-genome-size", default=500, type=int)
    sim_parser.add_argument("--init-molmap", default="randn", type=str)
    sim_parser.add_argument("--device", default="cpu", type=str)

    kwargs = vars(parser.parse_args())
    _cmds[kwargs.pop("cmd")](**kwargs)
