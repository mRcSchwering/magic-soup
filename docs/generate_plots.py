"""
Generate figures for docs/imgs/supporting

    PYTHONPATH=./python python docs/create_figures.py --help
"""

from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path

from plotnine import *  # pylint: disable=W0614,W0401

# from docs.plots.biochemical_patterns import create_plots as plot_biochemical_patterns
from docs.plots.free_energy import create_plots as plot_free_energy
from docs.plots.genomes import create_plots as plot_genomes
from docs.plots.molecule_physics import create_plots as plot_molecule_physics
from docs.plots.mutations import create_plots as plot_mutations
from docs.plots.param_distros import create_plots as plot_param_distros

# from docs.plots.reaction_kinetics import create_plots as plot_reaction_kinetics
from docs.plots.survival_replication import create_plots as plot_survival_replication
from docs.plots.transcriptomes import create_plots as plot_transcriptomes

_SUPPORTING_PLOTS: dict[str, Callable] = {
    "genomes": plot_genomes,
    "transcriptomes": plot_transcriptomes,
    "mutations": plot_mutations,
    "molecule-physics": plot_molecule_physics,
    "param-distros": plot_param_distros,
    #  "reaction_kinetics": plot_reaction_kinetics,
    #  "biochemical_patterns": plot_biochemical_patterns,
    "free-energy": plot_free_energy,
    "survival-replication": plot_survival_replication,
}


def generate_supporting_plots(plots: list[str] | None, not_transparent: bool):
    if plots is None or len(plots) == 0:
        plots = list(_SUPPORTING_PLOTS)

    imgs_dir = Path(__file__).parent.parent / "docs" / "img" / "supporting"
    if not imgs_dir.is_dir():
        raise ValueError(f"{imgs_dir} is not a directory")

    theme_set(theme_minimal())
    for name, fun in _SUPPORTING_PLOTS.items():
        if name in plots:
            print(f"plotting {name}...")
            fun(imgs_dir, not not_transparent)
            print(f"... done plotting {name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    supporting = subparsers.add_parser("supporting")
    supporting.add_argument("--plots", nargs="*", action="store")
    supporting.add_argument("--not-transparent", action="store_true")

    kwargs = vars(parser.parse_args())
    cmd = kwargs.pop("cmd")
    if cmd == "supporting":
        generate_supporting_plots(**kwargs)
