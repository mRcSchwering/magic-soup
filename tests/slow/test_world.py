import pytest
import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


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

    assert world.molecule_map.dtype is torch.float32
    assert world.cell_molecules.dtype is torch.float32
    assert world.cell_divisions.dtype is torch.int32
    assert world.cell_positions.dtype is torch.int32
    assert world.cell_lifetimes.dtype is torch.int32
    assert world.cell_map.dtype is torch.bool
