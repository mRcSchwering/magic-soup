from argparse import ArgumentParser
from contextlib import contextmanager
import time
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


@contextmanager
def timeit(msg: str):
    t0 = time.time()
    yield
    td = time.time() - t0
    print(f"{msg}: {td:.2f}s")


def main(n_cells: int, n_steps: int, init_genome_size: int):

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=REACTIONS)
    world = ms.World(chemistry=chemistry)

    genomes = [ms.random_genome(s=init_genome_size) for _ in range(n_cells)]
    genome_pairs = [(d, d) for d in genomes]

    with timeit(f"recombinations for {n_cells} genomes of size {init_genome_size}"):
        _ = ms.recombinations(seq_pairs=genome_pairs)

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_cells", default=1000, type=int)
    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--init_genome_size", default=500, type=int)
    args = parser.parse_args()

    main(
        n_cells=args.n_cells,
        n_steps=args.n_steps,
        init_genome_size=args.init_genome_size,
    )

