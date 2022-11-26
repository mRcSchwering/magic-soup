from enum import IntEnum
import torch
import numpy as np
from util import (
    ALL_NTS,
    CODON_SIZE,
    variants,
    reverse_complement,
    get_coding_regions,
    weight_map_fact,
)


class Signal(IntEnum):
    F = 0  # food
    CM = 1  # cell migration
    MA = 2  # messenger A
    MB = 3  # messenger B
    MC = 4  # messenger C
    MD = 5  # messenger D
    CK = 6  # cytokine K
    CL = 7  # cytokine L


# TODO: add protein-based params
# could add a constant to x and y (shift x or y axis)
# based on protein -> need another genetic expression for that
# imitates persistent expression, or reluctance of protein to express
# until certain threshold is overcome

# TODO: domains ideas
# - explicit oscillator (although transduction pathways can already become oscillators)
# - explicit memory/switch (transduction pathways could already become switches)

# domains: (name, signal, is_incomming)
DOMAINS: dict[tuple[str, Signal, bool], list[str]] = {
    ("RcF", Signal.F, True): variants("CTNTNN") + variants("CANANN"),
    ("RcK", Signal.CK, True): variants("CGNCNN") + variants("CGNANN"),
    ("RcL", Signal.CL, True): variants("AANCNN") + variants("ATNCNN"),
    ("ExK", Signal.CK, False): variants("ACNANN") + variants("ACNTNN"),
    ("ExL", Signal.CL, False): variants("TCNCNN") + variants("TANCNN"),
    ("Mig", Signal.CM, False): variants("GGNCNN") + variants("GTNCNN"),
    ("InA", Signal.MA, True): variants("CANGNN") + variants("CANCNN"),
    ("InB", Signal.MB, True): variants("CANTNN") + variants("CCNTNN"),
    ("InC", Signal.MC, True): variants("CGNGNN") + variants("CGNTNN"),
    ("InD", Signal.MD, True): variants("TTNGNN") + variants("TGNGNN"),
    ("OutA", Signal.MA, False): variants("TGNTNN") + variants("TANTNN"),
    ("OutB", Signal.MB, False): variants("GGNANN") + variants("GGNTNN"),
    ("OutC", Signal.MC, False): variants("CTNANN") + variants("CTNGNN"),
    ("OutD", Signal.MD, False): variants("TTNTNN") + variants("TTNANN"),
}
SEQ_2_DOM = {d: k for k, v in DOMAINS.items() for d in v}
DOMAIN_SIZE = 6  # with each 3 Ns in 2 codons ^= 3% chance of randomly appearing
WEIGHT_SIZE = 6  # 50% chance domain itself is mutated vs weight is mutated

CODON_2_RANGE0 = weight_map_fact(n_nts=WEIGHT_SIZE, mu=0, sd=1.3, is_positive=True)
CODON_2_RANGE1 = weight_map_fact(n_nts=WEIGHT_SIZE, mu=0, sd=1.3, is_positive=False)


def translate_seq(seq: str) -> dict[tuple[str, Signal, bool], float]:
    """
    Translate nucleotide sequence into dict that represents a protein
    with domains and corresponding weights.
    """
    i = 0
    j = DOMAIN_SIZE
    res = {}
    while j + WEIGHT_SIZE <= len(seq):
        dom = SEQ_2_DOM.get(seq[i:j])
        if dom is not None:
            range_map = CODON_2_RANGE1 if dom[-1] else CODON_2_RANGE0
            res[dom] = range_map[seq[j : j + WEIGHT_SIZE]]
            i += WEIGHT_SIZE + DOMAIN_SIZE
            j += WEIGHT_SIZE + DOMAIN_SIZE
        else:
            i += CODON_SIZE
            j += CODON_SIZE
    return res


def assert_config():
    """Check that configuration makes sense"""
    assert len(SEQ_2_DOM) == sum(
        len(d) for d in DOMAINS.values()
    ), "domain sequences overlapping"
    assert all(len(d) == DOMAIN_SIZE for d in SEQ_2_DOM), "domains have unequal sizes"
    assert all(
        set(d) <= set(ALL_NTS) for d in SEQ_2_DOM
    ), "domains include unknown nucleotides"


def get_proteome(
    g: str, ignore_cds=False
) -> list[dict[tuple[str, Signal, bool], float]]:
    """
    Get all possible proteins encoded by a nucleotide sequence.
    Proteins are represented as dicts with domain labels and correspondig
    weights.

    - `g` genome sequence (forward) 
    - `ignore_cds` whether to use the whole genome as coding sequence instead of
                   extracting all possible coding regions betwen start and stop codons

    21.11.22 time to get proteomes of 1000 generated genomes of size (1000, 5000)
    excluding genome generation time is 2.79s (mainly get_coding_regions, 0.43s
    without get_coding_regions)
    """
    gbwd = reverse_complement(g)
    if ignore_cds:
        cds = list(set([g, gbwd]))
    else:
        cds = list(set(get_coding_regions(g) + get_coding_regions(gbwd)))
    return [translate_seq(d) for d in cds]


def get_cell_params(
    cells: list[list[dict[tuple[str, Signal, bool], float]]]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate matrices A and B from cell proteomes

    Returns tuple (A, B) with matrices both of shape (c, s, p) where
    s is the number of signals, p is the number of proteins, and c is
    the number of cells.

    21.11.22 getting cell params for 1000 proteomes each of genome
    size (1000, 5000) took 0.01s
    """
    max_n_prots = max(len(d) for d in cells)
    n_cells = len(cells)
    n_sigs = len(Signal)
    a = np.zeros((n_cells, n_sigs, max_n_prots))
    b = np.zeros((n_cells, n_sigs, max_n_prots))
    for cell_i, cell in enumerate(cells):
        for prot_i, protein in enumerate(cell):
            for (_, sig, inc), weight in protein.items():
                sig_i = sig.value
                if inc:
                    a[cell_i, sig_i, prot_i] = weight
                else:
                    b[cell_i, sig_i, prot_i] = weight
    return (a, b)


def simulate_protein_work(C: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate molecules/signals created/activated by proteins after 1 timestep.
    Returns additional concentrations in shape `(c, s)`.

    - `C` initial concentrations shape `(c, s)`
    - `A` parameters for A `(c, s, p)`
    - `B` parameters for B `(c, s, p)`

    Proteins' activation function is `B * (1 - exp(-(C * A) ** 3))`.
    There are s signals, p proteins, c cells. The way how signals are
    integrated by multiplying with A we basically assume signals all
    activate and/or competitively inhibit the protein's domain.

    21.11.22 simulating protein work for 1000 cells each
    of genome size (1000, 5000) took 0.00s excluding other functions.
    """
    # matrix (c x p)
    x = np.einsum("ij,ijk->ik", C, A)
    y = 1 - np.exp(-(x ** 3))

    # matrix (c x m)
    return np.einsum("ij,ikj->ik", y, B)

