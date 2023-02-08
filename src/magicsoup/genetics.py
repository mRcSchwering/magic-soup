import warnings
import random
import torch.multiprocessing as mp
from magicsoup.util import reverse_complement, nt_seqs
from magicsoup.constants import CODON_SIZE


def _get_n(p: float, s: int, name: str) -> int:
    n = int(p * s)
    if n == 0 and p > 0.0:
        warnings.warn(
            f"There will be no {name}."
            f" Increase dom_type_size to accomodate low probabilities of having {name}."
        )
    return n


def _get_coding_regions(
    seq: str,
    min_cds_size: int,
    start_codons: tuple[str, ...],
    stop_codons: tuple[str, ...],
) -> list[str]:
    s = CODON_SIZE
    n = len(seq)
    max_start_idx = n - min_cds_size

    start_idxs = []
    for start_codon in start_codons:
        i = 0
        while i < max_start_idx:
            try:
                hit = seq[i:].index(start_codon)
                start_idxs.append(i + hit)
                i = i + hit + s
            except ValueError:
                break

    stop_idxs = []
    for stop_codon in stop_codons:
        i = 0
        while i < n - s:
            try:
                hit = seq[i:].index(stop_codon)
                stop_idxs.append(i + hit)
                i = i + hit + s
            except ValueError:
                break

    start_idxs.sort()
    stop_idxs.sort()

    by_frame: list[tuple[list[int], ...]] = [([], []), ([], []), ([], [])]
    for start_idx in start_idxs:
        if start_idx % 3 == 0:
            by_frame[0][0].append(start_idx)
        elif (start_idx + 1) % 3 == 0:
            by_frame[1][0].append(start_idx)
        else:
            by_frame[2][0].append(start_idx)
    for stop_idx in stop_idxs:
        if stop_idx % 3 == 0:
            by_frame[0][1].append(stop_idx)
        elif (stop_idx + 1) % 3 == 0:
            by_frame[1][1].append(stop_idx)
        else:
            by_frame[2][1].append(stop_idx)

    cdss = []
    for start_idxs, stop_idxs in by_frame:
        for start_idx in start_idxs:
            stop_idxs = [d for d in stop_idxs if d > start_idx + s]
            if len(stop_idxs) > 0:
                cds_end_idx = min(stop_idxs) + s
                if cds_end_idx - start_idx > min_cds_size:
                    cdss.append(seq[start_idx:cds_end_idx])
            else:
                break

    return cdss


def _extract_domains(
    cdss: list[str],
    dom_size: int,
    dom_type_size: int,
    dom_type_map: dict[str, int],
    one_codon_map: dict[str, int],
    two_codon_map: dict[str, int],
) -> list[list[tuple[int, int, int, int, int]]]:
    idx0_slice = slice(0, 2 * CODON_SIZE)
    idx1_slice = slice(2 * CODON_SIZE, 3 * CODON_SIZE)
    idx2_slice = slice(3 * CODON_SIZE, 4 * CODON_SIZE)
    idx3_slice = slice(4 * CODON_SIZE, 5 * CODON_SIZE)

    prot_doms = []
    for cds in cdss:
        doms = []
        is_useful_prot = False

        i = 0
        j = dom_size
        while i + dom_size <= len(cds):
            dom_type_seq = cds[i : i + dom_type_size]
            if dom_type_seq in dom_type_map:

                # 1=catal, 2=trnsp, 3=reg
                dom_type = dom_type_map[dom_type_seq]
                if dom_type != 3:
                    is_useful_prot = True

                dom_spec_seq = cds[i + dom_type_size : i + dom_size]
                idx0 = two_codon_map[dom_spec_seq[idx0_slice]]
                idx1 = one_codon_map[dom_spec_seq[idx1_slice]]
                idx2 = one_codon_map[dom_spec_seq[idx2_slice]]
                idx3 = one_codon_map[dom_spec_seq[idx3_slice]]
                doms.append((dom_type, idx0, idx1, idx2, idx3))
                i += dom_size
                j += dom_size
            else:
                i += CODON_SIZE
                j += CODON_SIZE

        # protein should have at least 1 non-regulatory domain
        if is_useful_prot:
            prot_doms.append(doms)

    return prot_doms


def _translate_genome(
    genome: str,
    dom_size: int,
    start_codons: tuple[str, ...],
    stop_codons: tuple[str, ...],
    dom_type_map: dict[str, int],
    one_codon_map: dict[str, int],
    two_codon_map: dict[str, int],
) -> list[list[tuple[int, int, int, int, int]]]:
    dom_type_size = len(next(iter(dom_type_map)))

    cdsf = _get_coding_regions(
        seq=genome,
        min_cds_size=dom_size,
        start_codons=start_codons,
        stop_codons=stop_codons,
    )
    bwd = reverse_complement(genome)
    cdsb = _get_coding_regions(
        seq=bwd,
        min_cds_size=dom_size,
        start_codons=start_codons,
        stop_codons=stop_codons,
    )

    prot_doms = _extract_domains(
        cdss=cdsf + cdsb,
        dom_size=dom_size,
        dom_type_size=dom_type_size,
        dom_type_map=dom_type_map,
        one_codon_map=one_codon_map,
        two_codon_map=two_codon_map,
    )

    return prot_doms


class Genetics:
    """
    Class holding logic about translating nucleotide sequences into proteomes.

    Arguments:
        start_codons: Start codons which start a coding sequence
        stop_codons: Stop codons which stop a coding sequence
        p_catal_dom: Chance of encountering a catalytic domain in a random nucleotide sequence.
        p_transp_dom: Chance of encountering a transporter domain in a random nucleotide sequence.
        p_reg_dom: Chance of encountering a regulatory domain in a random nucleotide sequence.
        n_dom_type_nts: Number of nucleotides that encodes the domain type (catalytic, transporter, regulatory).
        workers: number of workers

    During the simulation [World][magicsoup.world.World] uses [translate_genomes][magicsoup.genetics.Genetics.translate_genomes].
    The return value of this method is a nested list of tokens.
    These tokens are then mapped into concrete domain specifications (_e.g._ Km, Vmax, reactions, ...) by [Kinetics][magicsoup.kinetics.Kinetics].

    Translation happens only within coding sequences (CDSs).
    A CDS starts wherever a start codon is found and ends with the first in-frame encountered stop codon.
    Un-stopped CDSs are discarded.
    Both the forward and reverse complement of the nucleotide sequence are considered.
    Each CDS represents one protein.
    All domains found within a CDS will be added as domains to that protein.
    Unviable proteins, like proteins without domains or proteins with only regulatory domains, are discarded.

    If you want to use your own `genetics` object for the simulation you can just assign it after creating [World][magicsoup.world.World]:

    ```
        world = World(chemistry=chemistry)
        my_genetics = Genetics(p_transp_dom=0.1, stop_codons=("TGA", ))
        world.genetics = my_genetics
    ```

    Changing these attributes has a large effect on the information content of genomes.
    E.g. how many CDSs per nucleotide, i.e. how many proteins per nucleotide;
    how long are the average CDSs, i.e. how many domains per protein; how likely is it to encounter a domain;
    how many domains does each nucleotide encode, i.e. how likely is it for a single substitution to change the proteome,
    how likely is it that this mutation will only slightly change a domain or completely change it.
    """

    def __init__(
        self,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        p_catal_dom: float = 0.01,
        p_transp_dom: float = 0.01,
        p_reg_dom: float = 0.01,
        n_dom_type_nts: int = 6,
        workers: int = 2,
    ):
        if any(len(d) != CODON_SIZE for d in start_codons):
            raise ValueError(f"Not all start codons are of length {CODON_SIZE}")
        if any(len(d) != CODON_SIZE for d in stop_codons):
            raise ValueError(f"Not all stop codons are of length {CODON_SIZE}")
        if len(set(start_codons) & set(stop_codons)) > 0:
            raise ValueError(
                "Overlapping start and stop codons:"
                f" {','.join(str(d) for d in set(start_codons) & set(stop_codons))}"
            )
        if n_dom_type_nts % CODON_SIZE != 0:
            raise ValueError(f"n_dom_type_nts should be a multiple of {CODON_SIZE}.")

        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.workers = workers

        # Domain: domain_type (catal, trnsp, reg) + specification
        # specification: 1 x 2-codon token, 3 x 1-codon tokens
        # Domain can start and finish with start and stop codons, so this
        # is also the minimum CDS size
        self.dom_size = n_dom_type_nts + 5 * CODON_SIZE

        if p_catal_dom + p_transp_dom + p_reg_dom > 1.0:
            raise ValueError(
                "p_catal_dom, p_transp_dom, p_reg_dom together must not be greater 1.0"
            )

        sets = nt_seqs(n_dom_type_nts)
        random.shuffle(sets)
        n = len(sets)

        n_catal_doms = _get_n(p=p_catal_dom, s=n, name="catalytic domains")
        n_transp_doms = _get_n(p=p_transp_dom, s=n, name="transporter domains")
        n_reg_doms = _get_n(p=p_reg_dom, s=n, name="allosteric domains")

        # 1=catalytic, 2=transporter, 3=regulatory
        domain_types: dict[int, list[str]] = {}
        domain_types[1] = sets[:n_catal_doms]
        del sets[:n_catal_doms]
        domain_types[2] = sets[:n_transp_doms]
        del sets[:n_transp_doms]
        domain_types[3] = sets[:n_reg_doms]
        del sets[:n_reg_doms]

        self.domain_map = {d: k for k, v in domain_types.items() for d in v}

        self.two_codon_map: dict[str, int] = {}
        for i, seq in enumerate(nt_seqs(n=2 * CODON_SIZE)):
            self.two_codon_map[seq] = i + 1

        self.one_codon_map: dict[str, int] = {}
        for i, seq in enumerate(nt_seqs(n=CODON_SIZE)):
            self.one_codon_map[seq] = i + 1

    def translate_genomes(
        self, genomes: list[str]
    ) -> list[list[list[tuple[int, int, int, int, int]]]]:
        """
        Translate all genomes into proteomes

        Arguments:
            genomes: list of nucleotide sequences

        Returns:
            List of proteomes. This is a list (proteomes) of lists (proteins)
            of lists (domains) with each domain being a tuple of indices.
            These indices will be mapped to specific
            domain specifications by [Kinetics][magicsoup.kinetics.Kinetics].

        Both forward and reverse-complement are considered.
        CDSs are extracted and a protein is translated for every CDS.
        Unviable proteins (no domains or only regulatory domains) are discarded.
        """
        args = [
            (
                d,
                self.dom_size,
                self.start_codons,
                self.stop_codons,
                self.domain_map,
                self.one_codon_map,
                self.two_codon_map,
            )
            for d in genomes
        ]

        with mp.Pool(self.workers) as pool:
            dom_seqs = pool.starmap(_translate_genome, args)

        return dom_seqs
