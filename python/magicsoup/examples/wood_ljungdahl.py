"""
Wood-Ljungdahl Pathway

Using  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2646786/

Methyl (Eastern) Branch:
- formate dehydrogenase (1.2.1.43): CO2 + NADPH -> formiat + NADP (15 kJ/mol)
- formate-tetrahydrofolate ligase (6.3.4.3): formiat + FH4 + ATP -> formyl-FH4 + ADP + Pi (-5 kJ/mol)
- methenyltetrahydrofolate cyclohydrolase (3.5.4.9): formyl-FH4 + NADPH -> methylen-FH4 + NADP (-5 kJ/mol)
- methylenetetrahydrofolate reductase (1.1.99.15): methylen-FH4 + NADPH -> methyl-FH4 + NADP (-20 kJ/mol)

Carbonyl (Western) Branch:
- CFeSP methyltransferase (2.1.1.X): methyl-FH4 + Ni-ACS -> FH4 + methyl-Ni-ACS (-10 kJ/mol)
- CO dehydrogenase / acetyl-CoA synthetase: CO2 + HS-CoA + methyl-Ni-ACS -> Ni-ACS + acetyl-CoA (-40 kJ/mol)
- (skipped Co step)

Molecule energies were solved by solving for reaction energies exactly.
"""

from magicsoup.chemistry import Chemistry, Molecule, Reaction

NADPH = Molecule("NADPH", 62, half_life=3, diffusivity=4e-6, permeability=0)
NADP = Molecule("NADP", 2, half_life=60, diffusivity=4e-6, permeability=0)
ATP = Molecule("ATP", 33, half_life=210, diffusivity=2.5e-6, permeability=0)
ADP = Molecule("ADP", 3, half_life=500, diffusivity=2.5e-6, permeability=0)

methylFH4 = Molecule("methyl-FH4", 199, half_life=0.5, diffusivity=3e-6, permeability=1e-9)
methylenFH4 = Molecule("methylen-FH4", 159, half_life=0.1, diffusivity=3e-6, permeability=1e-9)
formylFH4 = Molecule("formyl-FH4", 104, half_life=0.02, diffusivity=3e-6, permeability=1e-9)
FH4 = Molecule("FH4", 2, half_life=0.02, diffusivity=3e-6, permeability=1e-9)
formiat = Molecule("formiat", 77, half_life=-1, diffusivity=1.5e-5, permeability=1e-6)
co2 = Molecule("CO2", 10, half_life=-1, diffusivity=2e-5, permeability=1)

NiACS = Molecule("Ni-ACS", 2, half_life=0.001, diffusivity=6e-7, permeability=0)
methylNiACS = Molecule("methyl-Ni-ACS", 189, half_life=0.001, diffusivity=6e-7, permeability=0)
HSCoA = Molecule("HS-CoA", 51, half_life=0.4, diffusivity=3.5e-6, permeability=0)
acetylCoA = Molecule("acetyl-CoA", 200, half_life=42, diffusivity=3.5e-6, permeability=0)


molecules = [
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

reactions = [
    Reaction([co2, NADPH], [formiat, NADP]),
    Reaction([formiat, FH4, ATP], [formylFH4, ADP]),
    Reaction([formylFH4, NADPH], [methylenFH4, NADP]),
    Reaction([methylenFH4, NADPH], [methylFH4, NADP]),
    Reaction([methylFH4, NiACS], [FH4, methylNiACS]),
    Reaction([methylNiACS, co2, HSCoA], [NiACS, acetylCoA]),
]

chemistry = Chemistry(molecules=molecules, reactions=reactions)
