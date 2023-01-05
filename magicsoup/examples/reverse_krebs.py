"""
free hand, energies invented, from
https://en.wikipedia.org/wiki/Reverse_Krebs_cycle
"""
from magicsoup.containers import Molecule, Chemistry

NADPH = Molecule("NADPH", 200.0)
NADP = Molecule("NADP", 100.0)
ATP = Molecule("ATP", 100.0)
ADP = Molecule("ADP", 70.0)
co2 = Molecule("CO2", 10.0)

oxalalcetate = Molecule("oxalalcetate", 200.0)
malate = Molecule("malate", 250.0)
fumarate = Molecule("fumarate", 240.0)
sucinate = Molecule("sucinate", 300.0)
sucinylCoA = Molecule("sucinyl-CoA", 500.0)
oxoglutarate = Molecule("oxoglutarate", 300.0)
isocitrate = Molecule("isocitrate", 350.0)
citrate = Molecule("citrate", 340.0)

HSCoA = Molecule("HS-CoA", 200.0)
acetylCoA = Molecule("acetyl-CoA", 260.0)

MOLECULES = [
    NADPH,
    NADP,
    ATP,
    ADP,
    co2,
    oxalalcetate,
    malate,
    fumarate,
    sucinate,
    sucinylCoA,
    oxoglutarate,
    isocitrate,
    citrate,
    HSCoA,
    acetylCoA,
]

REACTIONS = [
    ([oxalalcetate, NADPH], [malate, NADP]),
    ([malate], [fumarate]),
    ([fumarate, NADPH], [sucinate, NADP]),
    ([sucinate, ATP, HSCoA], [sucinylCoA, ADP]),
    ([sucinylCoA, co2], [oxoglutarate, HSCoA]),
    ([oxoglutarate, co2, NADPH], [isocitrate, NADP]),
    ([isocitrate], [citrate]),
    ([citrate, HSCoA], [acetylCoA, oxalalcetate]),
]

CHEMISTRY = Chemistry(molecules=MOLECULES, reactions=REACTIONS)
