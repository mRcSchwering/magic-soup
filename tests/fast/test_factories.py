import pytest
from magicsoup.constants import CODON_SIZE
import magicsoup as ms

_MI = ms.Molecule("i", 10 * 1e3)
_MJ = ms.Molecule("j", 20 * 1e3)
_MK = ms.Molecule("k", 30 * 1e3)


def test_generate_empty_genome():
    chemistry = ms.Chemistry(molecules=[_MI], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)

    ggen = ms.GenomeFact(world=world, proteome=[], target_size=10)
    g = ggen.generate()
    assert len(g) == 10
    ggen = ms.GenomeFact(world=world, proteome=[], target_size=100)
    g = ggen.generate()
    assert len(g) == 100

    ggen = ms.GenomeFact(world=world, proteome=[])
    g = ggen.generate()
    assert len(g) == 0


def test_generate_genome_different_sizes():
    molecules = [_MI, _MJ, _MK]
    reactions = [([_MI], [_MJ]), ([_MI, _MJ], [_MK])]

    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)

    p0 = [
        ms.CatalyticDomainFact(reaction=([_MI], [_MJ])),
        ms.CatalyticDomainFact(reaction=([_MK], [_MI, _MJ])),
    ]
    with pytest.raises(ValueError):
        ggen = ms.GenomeFact(world=world, proteome=[p0], target_size=10)

    ggen = ms.GenomeFact(world=world, proteome=[p0], target_size=100)
    g = ggen.generate()
    assert len(g) == 100

    ggen = ms.GenomeFact(world=world, proteome=[p0], target_size=None)
    g = ggen.generate()
    assert len(g) == 7 * CODON_SIZE * len(p0) + 2 * CODON_SIZE


def test_generate_correct_genome():
    molecules = [_MI, _MJ, _MK]
    reactions = [([_MI], [_MJ]), ([_MI, _MJ], [_MK])]

    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)

    p0 = [
        ms.CatalyticDomainFact(reaction=([_MI], [_MJ])),
        ms.CatalyticDomainFact(reaction=([_MK], [_MI, _MJ])),
    ]
    p1 = [
        ms.TransporterDomainFact(molecule=_MI),
        ms.RegulatoryDomainFact(effector=_MK, is_transmembrane=True, hill=3),
    ]
    ggen = ms.GenomeFact(world=world, proteome=[p0, p1], target_size=100)

    # codon-to-indices mappings (1-codon/2-codon maps) can also map to
    # stop codons, so there's a 4.6% chance per such codon that a domain
    # is pre-maturely terminated by a stop codon. With 5 such domains
    # in every domain, there's only a 78% chance that everything goes right
    # with 4 domains in the proteome, theres only a 38% chance of succeeding
    # (having 0 pre-mature terminations)
    # repeating the genome generation 10 times, reduces the chance of 10 pre-mature
    # terminations to below 1% (>95% chance that at least one proteome is correct)
    success = False
    max_i = 10
    i = 0
    p0_found = 0
    p1_found = 0
    while not success:
        g = ggen.generate()
        assert len(g) == 100

        world.spawn_cells(genomes=[g])
        cell = world.get_cell(by_idx=0)
        has_p0 = False
        has_p1 = False
        proteome = cell.proteome
        for prot in proteome:
            has_cij = False
            has_ckij = False
            has_ti = False
            has_rk = False

            for dom in prot.domains:
                if isinstance(dom, ms.CatalyticDomain):
                    subs = dom.substrates
                    prods = dom.products
                    if subs == [_MK] and prods == [_MI, _MJ]:
                        has_ckij = True
                    elif subs == [_MI] and prods == [_MJ]:
                        has_cij = True
                if isinstance(dom, ms.TransporterDomain):
                    if dom.molecule == _MI:
                        has_ti = True
                if isinstance(dom, ms.RegulatoryDomain):
                    if dom.effector == _MK and dom.is_inhibiting and dom.hill == 3:
                        has_rk = True

            if has_ckij and has_cij:
                has_p0 = True
                p0_found += 1
            if has_ti and has_rk:
                has_p1 = True
                p1_found += 1

        world.kill_cells(cell_idxs=list(range(world.n_cells)))
        if has_p0 and has_p1:
            success = True
            break
        else:
            i += 1

        if i > max_i:
            raise AssertionError(
                f"Was not able to recreate proteome from generated genome after {max_i} tries."
                f" P0 was found {p0_found} times, P1 was found {p1_found} times."
            )


def test_generate_genome_with_different_reaction_sorting():
    molecules = [_MI, _MJ, _MK]
    reactions = [([_MJ, _MI], [_MK])]

    # reaction gets sorted to i + j <-> k
    # when initializing chemistry object
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)

    # this should work because reaction was ordered
    doms = [ms.CatalyticDomainFact(reaction=reactions[0])]
    ggen = ms.GenomeFact(world=world, proteome=[doms], target_size=100)
    g = ggen.generate()
    assert len(g) == 100

    # in contrast this should fail,
    # because the reaction wasnt defined
    doms = [ms.CatalyticDomainFact(reaction=([_MJ], [_MK]))]
    with pytest.raises(ValueError):
        ggen = ms.GenomeFact(world=world, proteome=[doms], target_size=100)


def test_container_dcts_compatible_with_factories():
    molecules = [_MI, _MJ, _MK]
    reactions = [([_MJ, _MI], [_MK])]

    # reaction gets sorted to i + j <-> k
    # when initializing chemistry object
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)

    # this should work because reaction was ordered
    cdom = ms.CatalyticDomain(reaction=reactions[0], km=0.5, vmax=1.0, start=0, end=1)
    ms.CatalyticDomainFact.from_dict(cdom.to_dict())

    tdom = ms.TransporterDomain(
        molecule=molecules[0], km=0.5, vmax=1.0, is_exporter=True, start=2, end=3
    )
    ms.TransporterDomainFact.from_dict(tdom.to_dict())

    rdom = ms.RegulatoryDomain(
        effector=molecules[0],
        hill=3,
        km=0.5,
        is_inhibiting=True,
        is_transmembrane=True,
        start=4,
        end=5,
    )
    ms.RegulatoryDomainFact.from_dict(rdom.to_dict())

    prot = ms.Protein(domains=[cdom, tdom, rdom], cds_start=0, cds_end=1, is_fwd=True)
    ms.GenomeFact.from_dicts(dcts=[prot.to_dict()], world=world)
