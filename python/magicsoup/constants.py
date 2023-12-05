from itertools import product

CODON_SIZE = 3  # in number of nucleotides
GAS_CONSTANT = 8.31446261815324  # in J/(K*mol)

ALL_NTS = tuple("TCGA")  # "N" represents any one of these
ALL_CODONS = set("".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS))


# fmt: off
CODON_TABLE = {
    "TTT": 1,   "TCT": 2,   "TAT": 3,   "TGT": 4,
    "TTC": 5,   "TCC": 6,   "TAC": 7,   "TGC": 8,
    "TTA": 9,   "TCA": 10,  "TAA": 11,  "TGA": 12,
    "TTG": 13,  "TCG": 14,  "TAG": 15,  "TGG": 16,
    "CTT": 17,  "CCT": 18,  "CAT": 19,  "CGT": 20,
    "CTC": 21,  "CCC": 22,  "CAC": 23,  "CGC": 24,
    "CTA": 25,  "CCA": 26,  "CAA": 27,  "CGA": 28,
    "CTG": 29,  "CCG": 30,  "CAG": 31,  "CGG": 32,
    "ATT": 33,  "ACT": 34,  "AAT": 35,  "AGT": 36,
    "ATC": 37,  "ACC": 38,  "AAC": 39,  "AGC": 40,
    "ATA": 41,  "ACA": 42,  "AAA": 43,  "AGA": 44,
    "ATG": 45,  "ACG": 46,  "AAG": 47,  "AGG": 48,
    "GTT": 49,  "GCT": 50,  "GAT": 51,  "GGT": 52,
    "GTC": 53,  "GCC": 54,  "GAC": 55,  "GGC": 56,
    "GTA": 57,  "GCA": 58,  "GAA": 59,  "GGA": 60,
    "GTG": 61,  "GCG": 62,  "GAG": 63,  "GGG": 64
}
# fmt: on


DomainSpecType = tuple[tuple[int, int, int, int, int], int, int]
ProteinSpecType = tuple[list[DomainSpecType], int, int, bool]
