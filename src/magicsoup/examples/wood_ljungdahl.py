"""
using https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2646786/

Methyl (Eastern) Branch:
formate dehydrogenase (1.2.1.43): CO2 + NADPH -> formiat + NADP (- 21.5 kJ/mol)
formate-tetrahydrofolate ligase (6.3.4.3): formiat + FH4 + ATP -> formyl-FH4 + ADP + Pi (-8.4 kJ/mol)
methenyltetrahydrofolate cyclohydrolase (3.5.4.9): formyl-FH4 + NADPH -> methylen-FH4 + NADP (-40.2 kJ/mol)
methylenetetrahydrofolate reductase (1.1.99.15): methylen-FH4 + NADPH -> methyl-FH4 + NADP (-39.2 kJ/mol)

Carbonyl (Western) Branch:
CFeSP methyltransferase (2.1.1.X): methyl-FH4 + Ni-ACS -> FH4 + methyl-Ni-ACS
CO dehydrogenase / acetyl-CoA synthetase: CO2 + HS-CoA + methyl-Ni-ACS -> Ni-ACS + acetyl-CoA
(skipped Co step)
"""
from magicsoup.containers import Molecule, Chemistry

NADPH = Molecule("NADPH", 200.0 * 1e3)
NADP = Molecule("NADP", 100.0 * 1e3)
ATP = Molecule("ATP", 100.0 * 1e3)
ADP = Molecule("ADP", 70.0 * 1e3)

methylFH4 = Molecule("methyl-FH4", 360.0 * 1e3)
methylenFH4 = Molecule("methylen-FH4", 300.0 * 1e3)
formylFH4 = Molecule("formyl-FH4", 240.0 * 1e3)
FH4 = Molecule("FH4", 200.0 * 1e3)
formiat = Molecule("formiat", 20.0 * 1e3)
co2 = Molecule("CO2", 10.0 * 1e3, diffusivity=1.0, permeability=1.0)

NiACS = Molecule("Ni-ACS", 200.0 * 1e3)
methylNiACS = Molecule("methyl-Ni-ACS", 300.0 * 1e3)
HSCoA = Molecule("HS-CoA", 200.0 * 1e3)
acetylCoA = Molecule("acetyl-CoA", 260.0 * 1e3)


MOLECULES = [
    NADPH,
    NADP,
    ATP,
    ADP,
    methylFH4,
    methylenFH4,
    formylFH4,
    FH4,
    formiat,
    co2,
    NiACS,
    methylNiACS,
    HSCoA,
    acetylCoA,
]

REACTIONS = [
    ([co2, NADPH], [formiat, NADP]),
    ([formiat, FH4, ATP], [formylFH4, ADP]),
    ([formylFH4, NADPH], [methylenFH4, NADP]),
    ([methylenFH4, NADPH], [methylFH4, NADP]),
    ([methylFH4, NiACS], [FH4, methylNiACS]),
    ([methylNiACS, co2, HSCoA], [NiACS, acetylCoA]),
]

CHEMISTRY = Chemistry(molecules=MOLECULES, reactions=REACTIONS)
