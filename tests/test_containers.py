import warnings
import pytest
import magicsoup.containers as cntnrs


def test_same_molecules_get_same_instance():
    atp = cntnrs.Molecule(name="ATP", energy=10)
    atp2 = cntnrs.Molecule(name="ATP", energy=10)
    nadph = cntnrs.Molecule(name="NADPH", energy=100)
    nadph2 = cntnrs.Molecule(name="NADPH", energy=100)
    assert atp is atp2
    assert nadph is nadph2
    assert atp is not nadph


def test_raise_if_same_molecule_with_different_energy():
    _ = cntnrs.Molecule(name="ATP", energy=10)
    with pytest.raises(ValueError):
        cntnrs.Molecule(name="ATP", energy=20)


def test_similar_molecule_warning():
    _ = cntnrs.Molecule(name="ATP", energy=10)
    with warnings.catch_warnings(record=True) as warn:
        cntnrs.Molecule(name="atp", energy=10)
    assert len(warn) > 0
    assert issubclass(warn[-1].category, UserWarning)
