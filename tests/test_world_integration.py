import pytest
import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS
from tests.conftest import Retry


def test_molecule_amount_integrity_during_diffusion():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)

    exp = world.molecule_map.sum(dim=[1, 2])
    for step_i in range(100):
        world.diffuse_molecules()
        res = world.molecule_map.sum(dim=[1, 2])
        assert (res.sum() - exp.sum()).abs() < 10.0, step_i
        assert torch.all(torch.abs(res - exp) < 1.0), step_i


def test_molecule_amount_integrity_during_reactions():
    # X and Y can react back and forth but X + Y <-> Z
    # so if Z is counted as 2, n should stay equal
    mx = ms.Molecule("mx", 10 * 1e3)
    my = ms.Molecule("my", 20 * 1e3)
    mz = ms.Molecule("mz", 30 * 1e3)
    molecules = [mx, my, mz]
    reactions = [([mx], [my]), ([mx, my], [mz])]

    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    world.spawn_cells(genomes=genomes)

    def count(world: ms.World) -> float:
        mxy = world.molecule_map[[0, 1]].sum().item()
        mz = world.molecule_map[2].sum().item() * 2
        cxy = world.cell_molecules[:, [0, 1]].sum().item()
        cz = world.cell_molecules[:, 2].sum().item() * 2
        return mxy + mz + cxy + cz

    n0 = count(world)
    for step_i in range(100):
        world.enzymatic_activity()
        n = count(world)
        assert n == pytest.approx(n0, abs=1.0), step_i


def test_run_world_without_reactions():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry)

    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    world.spawn_cells(genomes=genomes)

    for _ in range(100):
        world.enzymatic_activity()


def test_exploding_molecules():
    # if e.g. during enzyme_activity calculation a reaction is not fair,
    # it can lead to cells generating molecules from nothing
    # this happens e.g. when X.clamp(min=0.0) is done before returning
    # if one side of the equation is slowed down, the other must be too
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=REACTIONS)
    world = ms.World(chemistry=chemistry, map_size=128)
    world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(1000)])

    for i in range(100):
        world.degrade_molecules()
        world.diffuse_molecules()
        world.enzymatic_activity()

        molmap = world.molecule_map
        cellmols = world.cell_molecules

        assert molmap.min() >= 0.0, i
        assert 0.0 < molmap.mean() < 50.0, i
        assert molmap.max() < 500.0, i

        assert cellmols.min() >= 0.0, i
        assert 0.0 < cellmols.mean() < 50.0, i
        assert cellmols.max() < 500.0, i


def test_genome_generation_consistency():
    n_tries = 5
    retry = Retry(n_allowed_fails=1)

    mi = ms.Molecule("mi", 10 * 1e3)
    mj = ms.Molecule("mj", 10 * 1e3)
    mk = ms.Molecule("mk", 10 * 1e3)
    molecules = [mi, mj, mk]
    reactions = [([mi], [mj]), ([mi, mj], [mk])]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry)

    # early stopping should not be possible
    # there is always the possibility that another protein
    # is created on the reverse-complement

    trsnp_dom = ms.TransporterDomainFact(
        molecule=mi, is_exporter=False, km=1.0, vmax=1.0
    )
    for i in range(n_tries):
        with retry.catch_assert(i):
            prtm = [ms.ProteinFact(domain_facts=trsnp_dom)]
            idxs = world.spawn_cells(
                genomes=[world.generate_genome(proteome=prtm, size=27)]
            )

            assert len(idxs) == 1
            ci = idxs[0]
            cell = world.get_cell(by_idx=ci)
            proteome = cell.get_proteome(world=world)
            assert len(proteome) == 1, proteome
            p0 = proteome[0]
            assert len(p0.domains) == 1, p0.domains
            d0 = p0.domains[0]

            assert isinstance(d0, ms.TransporterDomain)
            assert d0.molecule is mi
            assert abs(d0.vmax - 1.0) < 0.5
            assert abs(d0.km - 1.0) < 0.5
            assert not d0.is_exporter

            assert world.kinetics.N[ci][0][0] == 1, world.kinetics.N[ci]
            assert world.kinetics.N[ci][0][3] == -1, world.kinetics.N[ci]
            assert abs(world.kinetics.Vmax[ci][0] - 1.0) < 0.5
            assert abs(world.kinetics.Kmf[ci][0] - 1.0) < 0.5
            assert abs(world.kinetics.Kmb[ci][0] - 1.0) < 0.5

    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    retry.reset()

    catal_dom = ms.CatalyticDomainFact(reaction=([mj], [mi]), km=1.0, vmax=1.0)
    for i in range(n_tries):
        with retry.catch_assert(i):
            prtm = [ms.ProteinFact(domain_facts=catal_dom)]
            idxs = world.spawn_cells(
                genomes=[world.generate_genome(proteome=prtm, size=27)]
            )

            assert len(idxs) == 1
            ci = idxs[0]
            cell = world.get_cell(by_idx=ci)
            proteome = cell.get_proteome(world=world)
            assert len(proteome) == 1, proteome
            p0 = proteome[0]
            assert len(p0.domains) == 1
            d0 = p0.domains[0]

            assert isinstance(d0, ms.CatalyticDomain)
            assert d0.substrates[0] is mj
            assert d0.products[0] is mi
            assert abs(d0.vmax - 1.0) < 0.5
            assert abs(d0.km - 1.0) < 0.5

            assert world.kinetics.N[ci][0][0] == 1, world.kinetics.N[ci]
            assert world.kinetics.N[ci][0][1] == -1, world.kinetics.N[ci]
            assert abs(world.kinetics.Vmax[ci][0] - 1.0) < 0.5
            assert abs(world.kinetics.Kmf[ci][0] - 1.0) < 0.5
            assert abs(world.kinetics.Kmb[ci][0] - 1.0) < 0.5

    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    retry.reset()

    # with 2 domains there is a possibility that the last codon of the
    # 1st domain is a stop codon (=> a protein with only the 1st domain)
    # additionally, there could be a start codon in the first domain
    # (=> additional protein with only the second domain)
    # the reverse complement can always harbour another protein

    reg_dom = ms.RegulatoryDomainFact(
        effector=mk, is_transmembrane=True, is_inhibiting=True, km=1.0, hill=3
    )
    for i in range(n_tries):
        with retry.catch_assert(i):
            prtm = [ms.ProteinFact(domain_facts=[reg_dom, catal_dom])]
            idxs = world.spawn_cells(
                genomes=[world.generate_genome(proteome=prtm, size=48)]
            )

            assert len(idxs) == 1
            ci = idxs[0]
            cell = world.get_cell(by_idx=ci)
            proteome = cell.get_proteome(world=world)
            assert len(proteome) >= 1, proteome
            p0 = proteome[0]
            assert len(p0.domains) >= 2, p0.domains

            i = 0 if isinstance(p0.domains[0], ms.RegulatoryDomain) else 1
            assert isinstance(p0.domains[i], ms.RegulatoryDomain)
            assert p0.domains[i].effector is mk
            assert abs(p0.domains[i].km - 1.0) < 0.5
            assert p0.domains[i].is_inhibiting
            assert p0.domains[i].is_transmembrane
            assert p0.domains[i].hill == 3
            assert world.kinetics.A[ci, 0, 5] == -3, world.kinetics.A[ci]
            assert abs(world.kinetics.Kmr[ci, 0, 5] - 1.0) < 0.5
