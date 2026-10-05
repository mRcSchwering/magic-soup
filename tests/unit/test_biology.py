from magicsoup.biology import (
    CatalyticDomain,
    Protein,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.chemistry import Molecule

_mol_x = Molecule(name="X", energy=10)
_mol_y = Molecule(name="Y", energy=100)


def test_domains_from_to_dict():
    dct = {"reaction": (["X"], ["Y"]), "km": 1.0, "vmax": 2.0, "start": 1, "end": 2}
    dom = CatalyticDomain.from_dict(dct)
    assert dom.substrates == [_mol_x]
    assert dom.products == [_mol_y]
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
    dom = TransporterDomain.from_dict(dct)
    assert dom.molecule is _mol_x
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
    dom = RegulatoryDomain.from_dict(dct)
    assert dom.effector is _mol_x
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
    prot = Protein.from_dict(dct)
    assert prot.to_dict() == dct

    assert prot.cds_start == 1
    assert prot.cds_end == 2
    assert prot.is_fwd
    assert len(prot.domains) == 3
    assert prot.n_domains == 3

    dom = prot.domains[0]
    assert isinstance(dom, CatalyticDomain)
    assert dom.substrates == [_mol_x]
    assert dom.products == [_mol_y]
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2

    dom = prot.domains[2]
    assert isinstance(dom, TransporterDomain)
    assert dom.molecule is _mol_x
    assert dom.is_exporter
    assert dom.km == 1.0
    assert dom.vmax == 2.0
    assert dom.start == 1
    assert dom.end == 2

    dom = prot.domains[1]
    assert isinstance(dom, RegulatoryDomain)
    assert dom.effector is _mol_x
    assert dom.is_inhibiting
    assert dom.is_transmembrane
    assert dom.hill == 5
    assert dom.km == 1.0
    assert dom.start == 1
    assert dom.end == 2
