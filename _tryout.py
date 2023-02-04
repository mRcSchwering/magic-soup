import time
from argparse import ArgumentParser
import random
import torch

# import magicsoup as ms
# from magicsoup.examples.wood_ljungdahl import CHEMISTRY

CODON_SIZE = 3
MIN_CDS_SIZE = 12
START_CODONS = ("TTG", "GTG", "ATG")
STOP_CODONS = ("TGA", "TAG", "TAA")
NON_POLAR_AAS = ["F", "L", "I", "M", "V", "P", "A", "W", "G"]
POLAR_AAS = ["S", "T", "Y", "Q", "N", "C"]
BASIC_AAS = ["H", "K", "R"]
ACIDIC_AAS = ["D", "E"]

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
AA_MAP = {
    "Stop": 0,  # not defined
    "A": 1, "F": 2, "G": 3, "I": 4, "L": 5, "M": 6, "P": 7, "V": 8, "W": 9,  # non-polar 1-9
    "C": 10, "N": 11, "Q": 12, "S": 13, "T": 14, "Y": 15,  # polar 10-15
    "H": 16, "K": 17, "R": 18,  # basic 16-18
    "D": 19, "E": 20  # acidic 19-20
}
# fmt: on

# CODON_2_TOKEN = {k: AA_MAP[v] for k, v in CODON_TABLE.items()}
# TOKEN_2_AA = {v: k for k, v in CODON_2_TOKEN.items()}


# def _get_rnn() -> torch.nn.RNN:
#     w_ih = torch.tensor([[1.0, -100.0]])
#     w_hh = torch.tensor([[1.0]])
#     rnn = torch.nn.RNN(
#         input_size=2, hidden_size=1, num_layers=1, nonlinearity="relu", bias=False
#     )
#     rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=False)
#     rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=False)
#     return rnn


# @torch.no_grad()
# def _get_cumul_cdss(rnn: torch.nn.RNN, tokens: torch.Tensor) -> torch.Tensor:
#     starts = torch.isin(tokens, START_TOKENS).to(torch.float)
#     stops = torch.isin(tokens, STOP_TOKENS).to(torch.float)
#     X = torch.stack([starts, stops]).transpose(0, 2)  # seq, batch, feature
#     Y, _ = rnn(X)
#     return Y.squeeze(2).T  # batch, seq


# def _tokenize(genome: str) -> list[int]:
#     l = len(genome) + 1 - CODON_SIZE
#     ijs = [(i, i + CODON_SIZE) for i in range(0, l, CODON_SIZE)]
#     return [CODON_MAP[genome[i:j]] for i, j in ijs]


# def _tryout_coding_regions(
#     token_lst: list[list[int]], rnn: torch.nn.RNN
# ) -> torch.Tensor:
#     # create padded tokens tensor (n_sequences, max_l)
#     max_l = max(len(d) for d in token_lst)
#     tokens = torch.tensor([d + [0] * (max_l - len(d)) for d in token_lst])

#     # kind of cumsum that reverts to 0 if stop codon encountered
#     # every position > 0 is part of a CDS, but multiple CDSs can
#     # lay on top of each other, e.g. [0, 1, 1, 2, 2, 3, 3, 0, 0]
#     cumul_cdss = _get_cumul_cdss(rnn=rnn, tokens=tokens)

#     # each sequence has to be repeated by the maximum number of
#     # overlapping CDSs found, so that they can be disentangled
#     max_sums = cumul_cdss.amax(1).to(torch.int)
#     n_rows = int(max_sums.sum().item())

#     # do that repetition on tokens, cumul_cdss (n_rows, max_l)
#     expanded_tokens = tokens.repeat_interleave(max_sums, dim=0)
#     expanded_cumul_cdss = cumul_cdss.repeat_interleave(max_sums, dim=0)

#     # create incremental mask that marks CDSs: e.g. if we had
#     # [0, 1, 1, 2, 0], will check for >1: [0, 0, 0, 1, 0],
#     # and >0: [0, 1, 1, 1, 0], to find both CDSs
#     layers = torch.cat([torch.arange(d) for d in max_sums]).repeat(max_l, 1).T
#     cds_mask = expanded_cumul_cdss > layers

#     # find start and stop indexes of each marked CDS
#     pref = torch.zeros(n_rows, 1, dtype=torch.bool)
#     suf = torch.ones(n_rows, 1, dtype=torch.bool)
#     diff = torch.cat([cds_mask, suf], dim=1) != torch.cat([pref, cds_mask], dim=1)
#     idxs = torch.argwhere(diff)

#     # for each row find consequtive CDSs
#     cds_lst = []
#     for row_i in range(n_rows):
#         row_idxs = idxs[idxs[:, 0] == row_i, 1].tolist()

#         # zip behaviour ignores unstopped CDSs
#         for i, j in zip(row_idxs[:-1], row_idxs[1:]):
#             if j - i >= MIN_CDS_TOKENS:
#                 cds_lst.append(expanded_tokens[row_i, i:j])

#     cdss = torch.nn.utils.rnn.pad_sequence(cds_lst, batch_first=True)
#     return cdss


def timeit(callback, *args, r=3) -> tuple[float, float]:
    tds = []
    for _ in range(r):
        t0 = time.time()
        callback(*args)
        tds.append(time.time() - t0)
    m = sum(tds) / r
    s = sum((d - m) ** 2 / r for d in tds) ** (1 / 2)
    return m, s


# def classic_add_random_cells(world: ms.World, genomes: list[str]):
#     world.add_random_cells(genomes=genomes)


# def new_add_random_cells(world: ms.World, genomes: list[str]):
#     world.add_random_cells_new(genomes=genomes)


def dom_seq_to_tokens(seq: str) -> torch.Tensor:
    s = CODON_SIZE
    n = len(seq)
    ijs = [(i, i + s) for i in range(0, n + 1 - s, s)]
    tokens = [CODON_TABLE[seq[i:j]] for i, j in ijs]
    return torch.tensor(tokens)


def dom_seq_to_tokens2(seq: str) -> list[int]:
    s = CODON_SIZE
    n = len(seq)
    ijs = [(i, i + s) for i in range(0, n + 1 - s, s)]
    return [CODON_TABLE[seq[i:j]] for i, j in ijs]


def pad_r1(t: torch.Tensor, d, s: int) -> torch.Tensor:
    return torch.nn.functional.pad(
        t, pad=(0, 0, 0, s - t.size(0)), value=d, mode="constant"
    )


def pad_r2(t: torch.Tensor, d, s: int) -> torch.Tensor:
    return torch.nn.functional.pad(
        t, pad=(0, 0, 0, 0, 0, s - t.size(0)), value=d, mode="constant"
    )


def asd(dom_seqs_lst):
    n_prots = max(len(d) for d in dom_seqs_lst)
    n_doms = max(len(dd) for d in dom_seqs_lst for dd in d)

    c_doms = []
    for proteins in dom_seqs_lst:
        p_doms = []
        for doms in proteins:
            dom_seqs = []
            for seq in doms:
                dom_seqs.append(dom_seq_to_tokens(seq))
            p_doms.append(pad_r1(torch.stack(dom_seqs), s=n_doms, d=0))
        c_doms.append(pad_r2(torch.stack(p_doms), s=n_prots, d=0))
    tokens = torch.stack(c_doms)  # bool (c, p, d, n) n ^= domain length


def asd2(dom_seqs_lst):
    n_prots = max(len(d) for d in dom_seqs_lst)
    n_doms = max(len(dd) for d in dom_seqs_lst for dd in d)

    empty_dom = [0] * 10
    empty_prot = [empty_dom] * n_doms

    c_doms = []
    for proteins in dom_seqs_lst:
        p_doms = []
        for doms in proteins:
            dom_seqs = []
            for seq in doms:
                dom_seqs.append(dom_seq_to_tokens2(seq))
            p_doms.append(dom_seqs + [empty_dom] * (n_doms - len(dom_seqs)))
        c_doms.append(p_doms + [empty_prot] * (n_prots - len(p_doms)))
    tokens = torch.tensor(c_doms).size()  # bool (c, p, d, n) n ^= domain length


def main(n=1000, s=500, w=4):
    # print(f"{n:,} genomes, {s:,} size, {w} workers")
    # genomes = [ms.random_genome(s) for _ in range(n)]
    k = 10
    codons = list(CODON_TABLE)

    dom_seqs_lst = []
    for _ in range(n):
        prots = []
        for _ in range(random.randint(30, 50)):
            doms = []
            for _ in range(random.randint(1, 5)):
                doms.append("".join(random.choices(codons, k=k)))
            prots.append(doms)
        dom_seqs_lst.append(prots)

    mu, sd = timeit(asd, dom_seqs_lst)
    print(f"({mu:.2f}+-{sd:.2f})s - asd")

    mu, sd = timeit(asd2, dom_seqs_lst)
    print(f"({mu:.2f}+-{sd:.2f})s - asd2")

    # world = ms.World(chemistry=CHEMISTRY, workers=w)
    # mu, sd = timeit(classic_add_random_cells, world, genomes)
    # print(f"({mu:.2f}+-{sd:.2f})s - classic")

    # world = ms.World(chemistry=CHEMISTRY, workers=w)
    # mu, sd = timeit(new_add_random_cells, world, genomes)
    # print(f"({mu:.2f}+-{sd:.2f})s - new")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", default=1_000, type=int)
    parser.add_argument("--s", default=500, type=int)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    main(n=args.n, s=args.s, w=args.workers)
