import warnings
import pytest
from magicsoup.examples.wood_ljungdahl import REACTIONS, MOLECULES
import magicsoup.containers as cntnrs


@pytest.mark.parametrize(
    "doms1, doms2",
    [
        (
            [
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
                cntnrs.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
            ],
            [
                cntnrs.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                cntnrs.AllostericDomain(MOLECULES[0], 1.0, False, False),
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                cntnrs.AllostericDomain(MOLECULES[0], 1.0, False, False),
            ],
        ),
    ],
)
def test_comparing_same_proteins(doms1, doms2):
    p1 = cntnrs.Protein(domains=doms1, label="P1")
    p2 = cntnrs.Protein(domains=doms2, label="P2")
    assert p1 == p2


@pytest.mark.parametrize(
    "doms1, doms2",
    [
        (
            [
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 2.0, False),
                cntnrs.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
            ],
            [
                cntnrs.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                cntnrs.CatalyticDomain(REACTIONS[1], 1.0, 1.0, False),
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                cntnrs.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                cntnrs.AllostericDomain(MOLECULES[0], 1.0, True, False),
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                cntnrs.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                cntnrs.AllostericDomain(MOLECULES[0], 1.0, False, False),
            ],
        ),
    ],
)
def test_comparing_different_proteins(doms1, doms2):
    p1 = cntnrs.Protein(domains=doms1, label="P1")
    p2 = cntnrs.Protein(domains=doms2, label="P2")
    assert p1 != p2


def test_same_molecules_get_same_instance():
    xxx = cntnrs.Molecule(name="XXX", energy=10)
    xxx2 = cntnrs.Molecule(name="XXX", energy=10)
    yyy = cntnrs.Molecule(name="YYY", energy=100)
    yyy2 = cntnrs.Molecule(name="YYY", energy=100)
    assert xxx is xxx2
    assert yyy is yyy2
    assert yyy is not xxx


def test_raise_if_same_molecule_with_different_energy():
    _ = cntnrs.Molecule(name="XXX", energy=10)
    with pytest.raises(ValueError):
        cntnrs.Molecule(name="XXX", energy=20)


def test_similar_molecule_warning():
    _ = cntnrs.Molecule(name="XXX", energy=10)
    with warnings.catch_warnings(record=True) as warn:
        cntnrs.Molecule(name="xxx", energy=10)
    assert len(warn) > 0
    assert issubclass(warn[-1].category, UserWarning)
