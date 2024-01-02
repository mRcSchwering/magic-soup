# type: ignore
"""
Generate figures for docs/imgs/supporting

    PYTHONPATH=./python python docs/create_figures.py --help
"""
from typing import Callable
from pathlib import Path
from argparse import ArgumentParser
from plotnine import *
from plots.genomes import create_plots as plot_genomes
from plots.transcriptomes import create_plots as plot_transcriptomes
from plots.mutations import create_plots as plot_mutations
from plots.molecule_maps import create_plots as plot_molecule_maps
from plots.equilibrium_constants import create_plots as plot_equilibrium_constants
from plots.reaction_kinetics import create_plots as plot_reaction_kinetics
from plots.biochemical_patterns import create_plots as plot_biochemical_patterns
from plots.free_energy import create_plots as plot_free_energy
from plots.survival_replication import create_plots as plot_survival_replication
from plots.kinetic_constants import create_plots as plot_kinetic_constants


_PLOTS: dict[str, Callable] = {
    "genomes": plot_genomes,
    "transcriptomes": plot_transcriptomes,
    "mutations": plot_mutations,
    "molecule_maps": plot_molecule_maps,
    "equilibrium_constants": plot_equilibrium_constants,
    "reaction_kinetics": plot_reaction_kinetics,
    "biochemical_patterns": plot_biochemical_patterns,
    "free_energy": plot_free_energy,
    "survival_replication": plot_survival_replication,
    "kinetic_constants": plot_kinetic_constants,
}


def main(plots: list[str]):
    if len(plots) == 0:
        plots = list(_PLOTS)

    imgs_dir = Path(__file__).parent.parent / "docs" / "img" / "supporting"
    if not imgs_dir.is_dir():
        raise ValueError(f"{imgs_dir} is not a directory")

    theme_set(theme_minimal())
    for name, fun in _PLOTS.items():
        if name in plots:
            print(f"plotting {name}...")
            fun(imgs_dir)
            print(f"... done plotting {name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--plots", nargs="*", action="store")
    args = parser.parse_args()
    main(plots=args.plots)
