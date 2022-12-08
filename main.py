import time
import torch
import magicsoup as ms
from magicsoup.util import (
    rand_genome,
    variants,
    weight_map_fact,
    bool_map_fact,
    CODON_SIZE,
)
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


if __name__ == "__main__":
    # fmt: off
    domains = {
        ms.CatalyticFact(): variants("ACNTGN") + variants("AGNTGN") + variants("CCNTTN"),
        ms.TransporterFact(): variants("ACNAGN") + variants("ACNTAN") + variants("AANTCN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibitor=False): variants("GCNTGN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibitor=True): variants("GCNTAN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibitor=False): variants("AGNTCN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibitor=True): variants("CCNTGN"),
    }
    # fmt: on

    genetics = ms.Genetics(
        domain_facts=domains, molecules=MOLECULES, reactions=REACTIONS
    )

    world = ms.World(molecules=MOLECULES)

    g = rand_genome()
    p = genetics.get_proteome(seq=g)

    Km, Vmax, Ke, N, A = world.get_cell_params(proteomes=[p])
