from itertools import product

CODON_SIZE = 3
GAS_CONSTANT = 8.31446261815324  # J/K/mol
ALL_NTS = tuple("TCGA")
ALL_CODONS = tuple(set("".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS)))
