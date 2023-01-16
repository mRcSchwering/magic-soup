"""
free hand, energies invented, from
https://en.wikipedia.org/wiki/Reverse_Krebs_cycle
"""
from magicsoup.containers import Molecule, Chemistry

NADPH = Molecule("NADPH", 200.0 * 1e3)
NADP = Molecule("NADP", 100.0 * 1e3)
ATP = Molecule("ATP", 100.0 * 1e3)
ADP = Molecule("ADP", 70.0 * 1e3)
co2 = Molecule("CO2", 10.0 * 1e3, diffusivity=1.0, permeability=1.0)

oxalalcetate = Molecule("oxalalcetate", 200.0 * 1e3)
malate = Molecule("malate", 250.0 * 1e3)
fumarate = Molecule("fumarate", 240.0 * 1e3)
sucinate = Molecule("sucinate", 300.0 * 1e3)
sucinylCoA = Molecule("sucinyl-CoA", 500.0 * 1e3)
oxoglutarate = Molecule("oxoglutarate", 300.0 * 1e3)
isocitrate = Molecule("isocitrate", 350.0 * 1e3)
citrate = Molecule("citrate", 340.0 * 1e3)

HSCoA = Molecule("HS-CoA", 200.0 * 1e3)
acetylCoA = Molecule("acetyl-CoA", 260.0 * 1e3)

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
