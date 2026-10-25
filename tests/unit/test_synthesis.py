import pytest
from magicsoup.biology import (
    CatalyticDomain,
    Protein,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.chemistry import Molecule, Reaction
from magicsoup.examples.wood_ljungdahl import (
    ATP,
    CHEMISTRY,
    NADP,
    NADPH,
    co2,
    formiat,
    formylFH4,
)
from magicsoup.synthesis import (
    CatalyticDomainSynthesizer,
    GenomeSynthesizer,
    ProteinSynthesizer,
    RegulatoryDomainSynthesizer,
    TransporterDomainSynthesizer,
)
from magicsoup.world import World


@pytest.fixture
def world() -> World:
    return World(chemistry=CHEMISTRY)


def _is_correct(act: Protein, exp: ProteinSynthesizer) -> bool:
    exp_reacts = []
    exp_trnsps = []
    exp_effcts = []
    for exp_dom in exp.domains:
        if isinstance(exp_dom, CatalyticDomainSynthesizer):
            exp_reacts.append(exp_dom.reaction)
        if isinstance(exp_dom, TransporterDomainSynthesizer):
            exp_trnsps.append(exp_dom.molecule)
        if isinstance(exp_dom, RegulatoryDomainSynthesizer):
            exp_effcts.append(exp_dom.effector)

    for act_dom in act.domains:
        if isinstance(act_dom, CatalyticDomain) and act_dom.reaction in exp_reacts:
            exp_reacts.pop(exp_reacts.index(act_dom.reaction))
        if isinstance(act_dom, TransporterDomain) and act_dom.molecule in exp_trnsps:
            exp_trnsps.pop(exp_trnsps.index(act_dom.molecule))
        if isinstance(act_dom, RegulatoryDomain) and act_dom.effector in exp_effcts:
            exp_effcts.pop(exp_effcts.index(act_dom.effector))

    return all(len(d) == 0 for d in (exp_reacts, exp_trnsps, exp_effcts))


@pytest.mark.parametrize(
    "reacts, trnsps, effcts",
    [
        # no reaction
        ([], [NADPH], [ATP]),
        # no transport
        ([Reaction([co2, NADPH], [formiat, NADP])], [], [ATP]),
        # no regulatory
        ([], [NADPH], []),
        # multiple reactions
        ([Reaction([co2, NADPH], [formiat, NADP])] * 2, [], []),
        # multiple transporters
        ([], [formiat, formylFH4], []),
        # co-transport
        ([], [formiat, formiat], []),
        # multiple regulatory
        ([], [NADPH], [ATP, formiat]),
    ],
)
def test_generate_genomes(
    world: World, reacts: list[Reaction], trnsps: list[Molecule], effcts: list[Molecule]
) -> None:
    n_gens = 100

    p0 = ProteinSynthesizer(
        catalyzed_reactions=[Reaction([co2, NADPH], [formiat, NADP])],
        transported_molecules=[NADPH],
        allosteric_effectors=[ATP],
    )
    p1 = ProteinSynthesizer(
        catalyzed_reactions=reacts,
        transported_molecules=trnsps,
        allosteric_effectors=effcts,
    )

    synthesizer = GenomeSynthesizer(world=world, proteins=[p0, p1])

    genomes = [synthesizer.generate() for _ in range(n_gens)]

    # they are all different
    assert len(set(genomes)) == len(genomes)

    # each of them includes the desired proteins
    idxs = world.spawn_cells(genomes=genomes)

    correct_genomes = 0
    for idx in idxs.tolist():
        cell = world.get_cell(by_idx=idx)
        p0s = [d for d in cell.proteome if _is_correct(d, p0)]
        p1s = [d for d in cell.proteome if _is_correct(d, p1)]

        # proteome contains at least desired proteins
        if len(p0s) > 0 and len(p1s) > 0:
            correct_genomes += 1

    # TODO: for some reason slightly p0 is only encoded slightly less than 90%
    #       of the time correctly. When adding p1, together they are only correct
    #       slightly less than 70% -> investigate!
    assert correct_genomes / n_gens >= 0.6


@pytest.mark.parametrize("size", [100, 200])
def test_generate_genomes_with_custom_length(world: World, size: int) -> None:
    n_gens = 100

    p0 = ProteinSynthesizer(
        catalyzed_reactions=[Reaction([co2, NADPH], [formiat, NADP])],
        transported_molecules=[NADPH],
        allosteric_effectors=[ATP],
    )

    synthesizer = GenomeSynthesizer(world=world, proteins=[p0], target_size=size)

    genomes = [synthesizer.generate() for _ in range(n_gens)]

    # all have correct size
    assert all(len(d) == size for d in genomes)

    # they are all different
    assert len(set(genomes)) == len(genomes)

    # each of them includes the desired proteins
    idxs = world.spawn_cells(genomes=genomes)

    correct_genomes = 0
    for idx in idxs.tolist():
        cell = world.get_cell(by_idx=idx)
        p0s = [d for d in cell.proteome if _is_correct(d, p0)]

        # proteome contains at least desired proteins
        if len(p0s) > 0:
            correct_genomes += 1

    # TODO: for some reason slightly p0 is only encoded slightly less than 90%
    #       of the time correctly. -> investigate!
    assert correct_genomes / n_gens >= 0.8
