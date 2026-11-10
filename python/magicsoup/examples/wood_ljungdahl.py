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

Solve to get energies:

```
import z3

NADPH = z3.Int("NADPH")
NADP = z3.Int("NADP")
ATP = z3.Int("ATP")
ADP = z3.Int("ADP")
methylFH4 = z3.Int("methyl-FH4")
methylenFH4 = z3.Int("methylen-FH4")
formylFH4 = z3.Int("formyl-FH4")
FH4 = z3.Int("FH4")
formiat = z3.Int("formiat")
co2 = z3.Int("CO2")
NiACS = z3.Int("Ni-ACS")
methylNiACS = z3.Int("methyl-Ni-ACS")
HSCoA = z3.Int("HS-CoA")
acetylCoA = z3.Int("acetyl-CoA")

z3.solve(
    # non-zero
    NADPH > 1,
    NADP > 1,
    ATP > 1,
    ADP > 1,
    methylFH4 > 1,
    methylenFH4 > 1,
    formylFH4 > 1,
    FH4 > 1,
    formiat > 1,
    co2 > 1,
    NiACS > 1,
    methylNiACS > 1,
    HSCoA > 1,
    acetylCoA > 1,
    # reactions
    (formiat + NADP) - (co2 + NADPH) == 15,
    (formylFH4 + ADP) - (formiat + FH4 + ATP) == -5,
    (methylenFH4 + NADP) - (formylFH4 + NADPH) == -5,
    (methylFH4 + NADP) - (methylenFH4 + NADPH) == -20,
    (FH4 + methylNiACS) - (methylFH4 + NiACS) == -10,
    (NiACS + acetylCoA) - (co2 + HSCoA + methylNiACS) == -40,
    acetylCoA - HSCoA >= 30,
    ATP - ADP >= 30,
    NADPH - NADP >= 60,
    # other
    ADP > co2,
    ATP > ADP,
    NADPH > NADP,
    NADPH > ATP,
    formiat > co2,
    acetylCoA > HSCoA,
    formylFH4 > FH4,
    methylenFH4 > formylFH4,
    methylFH4 > methylenFH4,
    methylNiACS > NiACS,
    acetylCoA > methylNiACS,
    acetylCoA > methylFH4,
)
> [ATP = 33,
 FH4 = 2,
 methylen-FH4 = 159,
 formyl-FH4 = 104,
 NADPH = 62,
 acetyl-CoA = 200,
 HS-CoA = 51,
 methyl-Ni-ACS = 189,
 methyl-FH4 = 199,
 NADP = 2,
 Ni-ACS = 2,
 ADP = 3,
 CO2 = 2,
 formiat = 77]
```
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
    Reaction([co2, NADPH], [formiat, NADP]),
    Reaction([formiat, FH4, ATP], [formylFH4, ADP]),
    Reaction([formylFH4, NADPH], [methylenFH4, NADP]),
    Reaction([methylenFH4, NADPH], [methylFH4, NADP]),
    Reaction([methylFH4, NiACS], [FH4, methylNiACS]),
    Reaction([methylNiACS, co2, HSCoA], [NiACS, acetylCoA]),
]

CHEMISTRY = Chemistry(molecules=MOLECULES, reactions=REACTIONS)
