from itertools import product

EPS = 1e-4  # substrates don't get deconstructed below this amount
CODON_SIZE = 3  # in number of nucleotides
GAS_CONSTANT = 8.31446261815324  # in J/(K*mol)

ALL_NTS = tuple("TCGA")  # "N" represents any one of these
ALL_CODONS = tuple(set("".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS)))
