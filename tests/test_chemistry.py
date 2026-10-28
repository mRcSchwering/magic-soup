import pytest
from magicsoup.chemistry import Chemistry, Molecule, Reaction


def test_molecule_equality() -> None:
    m1 = Molecule("A", energy=10.0)
    m2 = Molecule("A", energy=10.0)
    m3 = Molecule("B", energy=15.0, half_life=10, diffusivity=2e-5, permeability=2e-10)

    assert m1 == m2
    assert m1 != m3


def test_reaction_equality() -> None:
    m1 = Molecule("A", energy=10.0)
    m2 = Molecule("B", energy=15.0)

    # all equal
    r1 = Reaction([m1, m1, m2], [m2])  # Reaction: 2A + B -> B
    r2 = Reaction([m1, m2, m1], [m2])  # Reaction: A + B + A -> B
    r3 = Reaction([m2], [m2] + 2 * [m1])  # Reaction: B -> B + 2A

    # not equal
    r4 = Reaction([m1, m2], [m2])  # Reaction: A + B -> B

    assert r1 == r2
    assert r2 == r3
    assert r3 != r4


def test_chemistry_initialization() -> None:
    m1 = Molecule("A", energy=10.0)
    m2 = Molecule("B", energy=15.0)
    m3 = Molecule("C", energy=20.0)

    # r1 ^= r2
    r1 = Reaction([m1], [m2])  # Reaction: A -> B
    r2 = Reaction([m2], [m1])  # Reaction: B -> A

    # r3 ^= r4 ^= r5
    r3 = Reaction([m1] * 3 + [m2], [m2])  # Reaction: 3A + B -> B
    r4 = Reaction([m2], [m1] * 3 + [m2])  # Reaction: B -> 3A + B
    r5 = Reaction([m1, m2, m1, m1], [m2])  # Reaction: A + B + 2A -> B

    chem = Chemistry(
        molecules=[m3, m2, m1, m2, m3],
        reactions=[r2, r3, r1, r4, r3, r2, r1, r5],
    )

    assert chem.molecules == [m1, m2, m3]

    # NOTE: order of reactions is not reliable
    #       since A <-> B == B <-> A
    assert set(chem.reactions) == {r1, r3}


def test_chemistry_raises_wrong_duplicate_molecule() -> None:
    m1 = Molecule("A", energy=10.0)
    m2 = Molecule("A", energy=20.0)  # Same name, different energy

    with pytest.raises(ValueError):
        Chemistry(molecules=[m1, m2], reactions=[])


def test_chemistry_raises_missing_molecule() -> None:
    m1 = Molecule("A", energy=10.0)
    m2 = Molecule("B", energy=15.0)
    m3 = Molecule("C", energy=20.0)

    r1 = Reaction([m1], [m2])  # Reaction: A -> B
    r2 = Reaction([m2], [m3])  # Reaction: B -> C

    with pytest.raises(ValueError):
        Chemistry(molecules=[m1, m2], reactions=[r1, r2])
