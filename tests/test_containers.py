import warnings
import pytest
import magicsoup.containers as cntnrs

X = cntnrs.Molecule(name="X", energy=10)
Y = cntnrs.Molecule(name="Y", energy=100)


def _get_domain(**kwargs) -> cntnrs.Domain:
    default_kwargs = {
        "substrates": [X],
        "products": [Y],
        "affinity": 0.1,
        "velocity": 1.0,
        "is_bkwd": False,
        "is_catalytic": True,
    }
    return cntnrs.Domain(**{**default_kwargs, **kwargs})  # type: ignore


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


def test_domain_comparisons():
    d1 = _get_domain()

    d2 = _get_domain(affinity=0.2)
    assert d1 != d2

    d2 = _get_domain(velocity=2.0)
    assert d1 != d2

    d2 = _get_domain(is_transmembrane=True)
    assert d1 != d2

    d2 = _get_domain(is_transporter=True)
    assert d1 != d2

    d2 = _get_domain(is_regulatory=True)
    assert d1 != d2

    d2 = _get_domain(is_inhibiting=True)
    assert d1 != d2

    d2 = _get_domain(products=[])
    assert d1 != d2

    d2 = _get_domain(substrates=[])
    assert d1 != d2

    d2 = _get_domain(is_bkwd=True)
    assert d1 != d2

