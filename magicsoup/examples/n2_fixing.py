"""

"""
from magicsoup.containers import Molecule, Chemistry

NADPH = Molecule("NADPH", 200.0)
NADP = Molecule("NADP", 100.0)
ATP = Molecule("ATP", 100.0)
ADP = Molecule("ADP", 70.0)

ammonia = Molecule("ammonia", 10)
glutamate = Molecule("glutamate", 200)
glutamine = Molecule("glutamine", 220)
oxalalcetate = Molecule("oxalalcetate", 200.0)

HSCoA = Molecule("HS-CoA", 200.0)
acetylCoA = Molecule("acetyl-CoA", 260.0)

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
