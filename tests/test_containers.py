import warnings
import pytest
import magicsoup.containers as cntnrs


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
