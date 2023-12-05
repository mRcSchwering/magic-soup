"""

"""
from magicsoup.containers import Molecule, Chemistry

NADPH = Molecule("NADPH", 200.0 * 1e3)
NADP = Molecule("NADP", 100.0 * 1e3)
ATP = Molecule("ATP", 100.0 * 1e3)
ADP = Molecule("ADP", 70.0 * 1e3)

ammonia = Molecule("ammonia", 10.0 * 1e3)
glutamate = Molecule("glutamate", 200.0 * 1e3)
glutamine = Molecule("glutamine", 220.0 * 1e3)
oxalalcetate = Molecule("oxalalcetate", 200.0 * 1e3)

HSCoA = Molecule("HS-CoA", 200.0 * 1e3)
acetylCoA = Molecule("acetyl-CoA", 260.0 * 1e3)

MOLECULES = [
    NADPH,
    NADP,
    ATP,
    ADP,
    ammonia,
    glutamate,
    glutamine,
    oxalalcetate,
    HSCoA,
    acetylCoA,
]

REACTIONS = [
    ([glutamate, ATP, ammonia], [ADP, glutamine]),
    ([oxalalcetate, glutamine, NADPH], [glutamate, glutamate, NADP]),
]

CHEMISTRY = Chemistry(molecules=MOLECULES, reactions=REACTIONS)
