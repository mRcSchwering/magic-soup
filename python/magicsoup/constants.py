from itertools import product

CODON_SIZE = 3  # in number of nucleotides
GAS_CONSTANT = 8.31446261815324  # in J/(K*mol)

ALL_NTS = tuple("TCGA")  # "N" represents any one of these
ALL_CODONS = {"".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS)}

type DomainSpecType = tuple[tuple[int, int, int, int, int], int, int]
type ProteinSpecType = tuple[list[DomainSpecType], int, int, bool]


# TODO: absolute temperature to room temp? easier diffusivity numbers

# TODO: add constants here

# TODO: use DTYPE constants as default values (no need to always set them)
