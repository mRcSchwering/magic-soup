from itertools import product

CODON_SIZE = 3  # in number of nucleotides
GAS_CONSTANT = 8.31446261815324  # in J/(K*mol)

ALL_NTS = tuple("TCGA")  # "N" represents any one of these
ALL_CODONS = set("".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS))

DomainSpecType = tuple[tuple[int, int, int, int, int], int, int]
ProteinSpecType = tuple[list[DomainSpecType], int, int, bool]
