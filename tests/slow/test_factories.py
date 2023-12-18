import magicsoup as ms
from tests.conftest import Retry


def test_genome_generation_consistency():
    n_tries = 6
    retry = Retry(n_allowed_fails=3)
    km_tol = 5.0
    vmax_tol = 1.0

    mi = ms.Molecule("mi", 10 * 1e3)
    mj = ms.Molecule("mj", 10 * 1e3)
    mk = ms.Molecule("mk", 10 * 1e3)
    molecules = [mi, mj, mk]
    reactions = [([mi], [mj]), ([mi, mj], [mk])]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry)

    # early stopping should not be possible but
    # there is always the possibility that another protein
    # is created on the reverse-complement

    dt = ms.TransporterDomainFact(molecule=mi, is_exporter=False, km=1.0, vmax=1.0)
    ggen = ms.GenomeFact(world=world, proteome=[[dt]])
    for i in range(n_tries):
        with retry.catch_assert(i):
            idxs = world.spawn_cells(genomes=[ggen.generate()])

            assert len(idxs) == 1
            ci = idxs[0]
            cell = world.get_cell(by_idx=ci)
            assert len(cell.proteome) == 1, cell.proteome
            p0 = cell.proteome[0]
            assert len(p0.domains) == 1, p0.domains
            d0 = p0.domains[0]

            assert isinstance(d0, ms.TransporterDomain)
            assert d0.molecule is mi
            assert abs(d0.vmax - 1.0) < vmax_tol
            assert abs(d0.km - 1.0) < km_tol
            assert not d0.is_exporter

            assert world.kinetics.N[ci][0][0] == 1, world.kinetics.N[ci]
            assert world.kinetics.N[ci][0][3] == -1, world.kinetics.N[ci]
            assert abs(world.kinetics.Vmax[ci][0] - 1.0) < vmax_tol
            assert abs(world.kinetics.Kmf[ci][0] - 1.0) < km_tol
            assert abs(world.kinetics.Kmb[ci][0] - 1.0) < km_tol

    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    retry.reset()

    dc = ms.CatalyticDomainFact(reaction=([mj], [mi]), km=1.0, vmax=1.0)
    ggen = ms.GenomeFact(world=world, proteome=[[dc]])
    for i in range(n_tries):
        with retry.catch_assert(i):
            idxs = world.spawn_cells(genomes=[ggen.generate()])

            assert len(idxs) == 1
            ci = idxs[0]
            cell = world.get_cell(by_idx=ci)
            assert len(cell.proteome) == 1, cell.proteome
            p0 = cell.proteome[0]
            assert len(p0.domains) == 1
            d0 = p0.domains[0]

            assert isinstance(d0, ms.CatalyticDomain)
            assert d0.substrates[0] is mj
            assert d0.products[0] is mi
            assert abs(d0.vmax - 1.0) < vmax_tol
            assert abs(d0.km - 1.0) < km_tol

            assert world.kinetics.N[ci][0][0] == 1, world.kinetics.N[ci]
            assert world.kinetics.N[ci][0][1] == -1, world.kinetics.N[ci]
            assert abs(world.kinetics.Vmax[ci][0] - 1.0) < vmax_tol
            assert abs(world.kinetics.Kmf[ci][0] - 1.0) < km_tol
            assert abs(world.kinetics.Kmb[ci][0] - 1.0) < km_tol

    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    retry.reset()

    # with 2 domains there is a possibility that the last codon of the
    # 1st domain is a stop codon (=> a protein with only the 1st domain)
    # additionally, there could be a start codon in the first domain
    # (=> additional protein with only the second domain)
    # the reverse complement can always harbour another protein

    dr = ms.RegulatoryDomainFact(
        effector=mk, is_transmembrane=True, is_inhibiting=True, km=1.0, hill=3
    )
    ggen = ms.GenomeFact(world=world, proteome=[[dr, dc]])
    for i in range(n_tries):
        with retry.catch_assert(i):
            idxs = world.spawn_cells(genomes=[ggen.generate()])

            assert len(idxs) == 1
            ci = idxs[0]
            cell = world.get_cell(by_idx=ci)
            assert len(cell.proteome) > 0

            pi = -1
            for i, prot in enumerate(cell.proteome):
                for dom in prot.domains:
                    if isinstance(dom, ms.RegulatoryDomain):
                        assert dom.effector is mk
                        assert abs(dom.km - 1.0) < km_tol
                        assert dom.is_inhibiting
                        assert dom.is_transmembrane
                        assert dom.hill == 3
                        pi = i

            assert pi > -1
            assert world.kinetics.A[ci, pi, 5] == -3, world.kinetics.A[ci]
            assert abs(world.kinetics.Kmr[ci, pi, 5] ** (-3) - 1.0) < km_tol
