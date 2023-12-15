import warnings
import pytest
import magicsoup.containers as cntnrs

_X = cntnrs.Molecule(name="X", energy=10)
_Y = cntnrs.Molecule(name="Y", energy=100)


def test_same_molecules_get_same_instance():
    X2 = cntnrs.Molecule(name="X", energy=10)
    Y2 = cntnrs.Molecule(name="Y", energy=100)
    assert _X is X2
    assert _Y is Y2
    assert _Y is not _X


def test_raise_if_same_molecule_with_different_energy():
    with pytest.raises(ValueError):
        cntnrs.Molecule(name="X", energy=20)


def test_similar_molecule_warning():
    with warnings.catch_warnings(record=True) as warn:
        cntnrs.Molecule(name="x", energy=10)
    assert len(warn) > 0
    assert issubclass(warn[-1].category, UserWarning)


def test_molecule_mappings():
    chem = cntnrs.Chemistry(molecules=[_X, _Y], reactions=[])
    assert chem.mol_2_idx[_X] == 0
    assert chem.mol_2_idx[_Y] == 1
    assert chem.molname_2_idx["X"] == 0
    assert chem.molname_2_idx["Y"] == 1


def test_molecule_from_name():
    X2 = cntnrs.Molecule.from_name(name="X")
    Y2 = cntnrs.Molecule.from_name(name="Y")
    assert X2 is _X
    assert Y2 is _Y


def test_domains_from_dict():
    kwargs = {"reaction": (["X"], ["Y"]), "km": 1.0, "vmax": 2.0, "start": 1, "end": 2}
    dom = cntnrs.CatalyticDomain.from_dict(kwargs)
    assert dom.substrates == [_X]
    assert dom.products == [_Y]
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2

    kwargs = {
        "molecule": "X",
        "km": 1.0,
        "vmax": 2.0,
        "is_exporter": True,
        "start": 1,
        "end": 2,
    }
    dom = cntnrs.TransporterDomain.from_dict(kwargs)
    assert dom.molecule is _X
    assert dom.is_exporter
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2

    kwargs = {
        "effector": "X",
        "km": 1.0,
        "hill": 5,
        "is_inhibiting": True,
        "is_transmembrane": True,
        "start": 1,
        "end": 2,
    }
    dom = cntnrs.RegulatoryDomain.from_dict(kwargs)
    assert dom.effector is _X
    assert dom.is_inhibiting
    assert dom.is_transmembrane
    assert dom.hill == 5
    assert dom.km == 1.0
    assert dom.start == 1
    assert dom.end == 2
