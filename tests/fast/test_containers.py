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


def test_domains_from_to_dict():
    dct = {"reaction": (["X"], ["Y"]), "km": 1.0, "vmax": 2.0, "start": 1, "end": 2}
    dom = cntnrs.CatalyticDomain.from_dict(dct)
    assert dom.substrates == [_X]
    assert dom.products == [_Y]
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2
    dct2 = dom.to_dict()
    assert dct2["spec"] == dct
    assert dct2["type"] == "C"

    dct = {
        "molecule": "X",
        "km": 1.0,
        "vmax": 2.0,
        "is_exporter": True,
        "start": 1,
        "end": 2,
    }
    dom = cntnrs.TransporterDomain.from_dict(dct)
    assert dom.molecule is _X
    assert dom.is_exporter
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2
    dct2 = dom.to_dict()
    assert dct2["spec"] == dct
    assert dct2["type"] == "T"

    dct = {
        "effector": "X",
        "km": 1.0,
        "hill": 5,
        "is_inhibiting": True,
        "is_transmembrane": True,
        "start": 1,
        "end": 2,
    }
    dom = cntnrs.RegulatoryDomain.from_dict(dct)
    assert dom.effector is _X
    assert dom.is_inhibiting
    assert dom.is_transmembrane
    assert dom.hill == 5
    assert dom.km == 1.0
    assert dom.start == 1
    assert dom.end == 2
    dct2 = dom.to_dict()
    assert dct2["spec"] == dct
    assert dct2["type"] == "R"


def test_protein_from_to_dict():
    cat_dct = {
        "type": "C",
        "spec": {
            "reaction": (["X"], ["Y"]),
            "km": 1.0,
            "vmax": 2.0,
            "start": 1,
            "end": 2,
        },
    }
    trnsp_dct = {
        "type": "T",
        "spec": {
            "molecule": "X",
            "km": 1.0,
            "vmax": 2.0,
            "is_exporter": True,
            "start": 1,
            "end": 2,
        },
    }
    reg_dct = {
        "type": "R",
        "spec": {
            "effector": "X",
            "km": 1.0,
            "hill": 5,
            "is_inhibiting": True,
            "is_transmembrane": True,
            "start": 1,
            "end": 2,
        },
    }

    dct = {
        "cds_start": 1,
        "cds_end": 2,
        "is_fwd": True,
        "domains": [cat_dct, reg_dct, trnsp_dct],
    }
    prot = cntnrs.Protein.from_dict(dct)
    assert prot.to_dict() == dct

    assert prot.cds_start == 1
    assert prot.cds_end == 2
    assert prot.is_fwd
    assert len(prot.domains) == 3
    assert prot.n_domains == 3

    dom = prot.domains[0]
    assert isinstance(dom, cntnrs.CatalyticDomain)
    assert dom.substrates == [_X]
    assert dom.products == [_Y]
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2

    dom = prot.domains[2]
    assert isinstance(dom, cntnrs.TransporterDomain)
    assert dom.molecule is _X
    assert dom.is_exporter
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2

    dom = prot.domains[1]
    assert isinstance(dom, cntnrs.RegulatoryDomain)
    assert dom.effector is _X
    assert dom.is_inhibiting
    assert dom.is_transmembrane
    assert dom.hill == 5
    assert dom.km == 1.0
    assert dom.start == 1
    assert dom.end == 2
