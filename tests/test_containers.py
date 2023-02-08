import warnings
import pytest
import magicsoup.containers as cntnrs

X = cntnrs.Molecule(name="X", energy=10)
Y = cntnrs.Molecule(name="Y", energy=100)


def test_same_molecules_get_same_instance():
    X2 = cntnrs.Molecule(name="X", energy=10)
    Y2 = cntnrs.Molecule(name="Y", energy=100)
    assert X is X2
    assert Y is Y2
    assert Y is not X


def test_raise_if_same_molecule_with_different_energy():
    with pytest.raises(ValueError):
        cntnrs.Molecule(name="X", energy=20)


def test_similar_molecule_warning():
    with warnings.catch_warnings(record=True) as warn:
        cntnrs.Molecule(name="x", energy=10)
    assert len(warn) > 0
    assert issubclass(warn[-1].category, UserWarning)
