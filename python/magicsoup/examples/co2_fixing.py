"""
# CO2 fixation pathways

- Calvin cycle
- 3-hydroxypropionate cycle
- Wood-Ljungdahl cycle
- reductive TCA cycle
- dicarboxylate/4-hydroxybutyrate cycle
- 3-hydroxypropionate/4-hydroxybutyrate cycle

from Gong, Fuyu & Cai, Zhen & Li, Yin. (2016). Synthetic biology for CO2 fixation,
https://en.wikipedia.org/wiki/Biological_carbon_fixation#Overview%20of%20pathways, and https://equilibrator.weizmann.ac.il
Some reactions are combined for simplicity. Common intermediates between different pathways are kept to allow cell to use combinations of pathways.
Molecule energies were solved using z3 by minimizing the absolute difference between expected reaction energies and calculated reaction energies
while only allowing a maximum per reaction error of 15 kJ/mol.
Diffusion coefficients were derived from charge and size using CO2, CO, Glucose, Sucrose, NADH, CoA as anchor points.
Permeability was deemed neglectable for all but CO2, CO, Q, QH2.
Half-lifes were assigned to the molecule's most liable group, neglecting everything above 100 days.

## Calvin cycle

Stage 1:
RuBP + CO2 -> 2 3PGA (-40 kJ/mol)
3PGA + ATP -> 1,3BPG + ADP (20 kJ/mol)
1,3BPG + NADPH -> G3P + NADP (-31 kJ/mol)

Stage 2:
5 G3P -> 3 Ru5P (0 kJ/mol)
Ru5P + ATP -> RuBP + ADP (-25 kJ/mol)

fixed carbon: G3P
chemical energy: ATP
reducing power: NADPH
seed: RuBP

## 3-Hydroxypropionate Bicycle

CO2 fixation:
acetyl-CoA + CO2 + ATP -> malonyl-CoA + ADP + Pi (-25 kJ/mol)
malonyl-CoA + 3 NADPH + ATP -> propionyl-CoA + 3 NADP + AMP + PPi (-50 kJ/mol)
propionyl-CoA + CO2 + ATP -> methylmalonyl-CoA + ADP + Pi (-25 kJ/mol)
methylmalonyl-CoA -> succinyl-CoA (0 kJ/mol)
succinyl-CoA -> succinate + HS-CoA (-35 kJ/mol)
succinate + Q -> fumarate + QH2 (-10 kJ/mol)
fumarate -> malate (-4 kJ/mol)
malate + HS-CoA -> malyl-CoA (35 kJ/mol)
malyl-CoA -> acetyl-CoA + glyoxylate (-5 kJ/mol)

Glyoxylate Assimilation:
propionyl-CoA + glyoxylate -> methylmalyl-CoA (-5 kJ/mol)
methylmalyl-CoA -> citramalyl-CoA (0 kJ/mol)
citramalyl-CoA -> acetyl-CoA + pyruvate (-10 kJ/mol)

fixed carbon: pyruvate
chemical energy: ATP
reducing power: NADPH, QH2
seed: acetyl-CoA

## Wood-Ljungdahl cycle

Methyl (Eastern) Branch:
CO2 + NADPH -> formate + NADP (0 kJ/mol)
formate + FH4 + ATP -> formyl-FH4 + ADP + Pi (-30 kJ/mol)
formyl-FH4 + NADPH -> methylen-FH4 + NADP (-15 kJ/mol)
methylen-FH4 + NADPH -> methyl-FH4 + NADP (-23 kJ/mol)

Carbonyl (Western) Branch:
CO2 + 2 Fe-red -> CO + 2 Fe-ox (0 kJ/mol)
methyl-FH4 + CO + HS-CoA -> acetyl-CoA + FH4 (-20 kJ/mol)

fixed carbon: acetyl-CoA
chemical energy: ATP
reducing power: NADPH, Fe-red
seed: acetyl-CoA
cofactors: FH4, HS-CoA

## Reductive TCA

oxalacetate + NADPH -> malate + NADP (-30 kJ/mol)
malate -> fumarate (4 kJ/mol)
fumarate + FADH2 -> succinate + FAD (-30 kJ/mol)
succinate + HS-CoA + ATP -> succinyl-CoA + ADP + Pi (2 kJ/mol)
succinyl-CoA + 2 Fd-red + CO2 -> a-ketoglutarate + HS-CoA + 2 Fd-ox (5 kJ/mol)
a-ketoglutarate + CO2 + NADPH -> isocitrate + NADP (+12 kJ/mol)
isocitrate -> citrate (-14 kJ/mol)
citrate + HS-CoA + ATP -> oxalacetate + acetyl-CoA + ADP + Pi (-8 kJ/mol)

fixed carbon: acetyl-CoA
chemical energy: ATP
reducing power: NADPH, FADH2, Fd-red
seed: oxalacetate
cofactor: HS-CoA

## Dicarboxylate/4-Hydroxybutyrate Cycle

acetyl-CoA + CO2 + 2 Fd-red -> pyruvate + 2 Fd-ox + HS-CoA (0 kJ/mol)
pyruvate + ATP + Pi -> PEP + AMP + PPi (-30 kJ/mol)
PEP + CO2 -> oxalacetate + Pi (-30 kJ/mol)
oxalacetate + NADPH -> malate + NADP (-30 kJ/mol)
malate -> fumarate (4 kJ/mol)
fumarate + 2 Fd-red -> succinate + 2 Fd-ox (-90 kJ/mol)
succinate + HS-CoA + ATP -> succinyl-CoA + ADP + Pi (2 kJ/mol)
succinyl-CoA + 2 Fd-red -> succinic-semialdehyde + HS-CoA + 2 Fd-ox (-15 kJ/mol)
succinic-semialdehyde + NADPH -> 4-hydroxy-butyrate + NADP (-25 kJ/mol)
4-hydroxy-butyrate + HS-CoA + ATP -> hydroxybutyryl-CoA + AMP + PPi (-35 kJ/mol)
hydroxybutyryl-CoA + NADP -> acetoacetyl-CoA + NADPH (10 kJ/mol)
acetoacetyl-CoA + HS-CoA -> 2 acetyl-CoA (0 kJ/mol)

fixed carbon: acetyl-CoA
chemical energy: ATP
reducing power: Fd-red, NADPH
seed: acetyl-CoA

## 3-Hydroxypropionate/4-Hydroxybutyrate Cycle

acetyl-CoA + CO2 + ATP -> malonyl-CoA + ADP + Pi (-25 kJ/mol)
malonyl-CoA + 3 NADPH + ATP -> propionyl-CoA + 3 NADP + AMP + PPi (-50 kJ/mol)
propionyl-CoA + CO2 + ATP -> methylmalonyl-CoA + ADP + Pi (-25 kJ/mol)
methylmalonyl-CoA -> succinyl-CoA (0 kJ/mol)
succinyl-CoA + NADPH -> succinic-semialdehyde + HS-CoA + NADP (5 kJ/mol)
succinic-semialdehyde + NADPH -> 4-hydroxy-butyrate + NADP (-25 kJ/mol)
4-hydroxy-butyrate + HS-CoA + ATP -> hydroxybutyryl-CoA + AMP + PPi (-35 kJ/mol)
hydroxybutyryl-CoA + NADP -> acetoacetyl-CoA + NADPH (10 kJ/mol)
acetoacetyl-CoA + HS-CoA -> 2 acetyl-CoA (0 kJ/mol)

fixed carbon: acetyl-CoA
chemical energy: ATP
reducing power: NADPH
seed: acetyl-CoA
"""

from magicsoup.chemistry import Chemistry, Molecule, Reaction

_AMP = Molecule("AMP", 1, diffusivity=3e-6)
_fumarate = Molecule("fumarate", 412, diffusivity=6e-6)
_isocitrate = Molecule("isocitrate", 632, diffusivity=5e-6)
_citramalylCoA = Molecule("citramalyl-CoA", 628, diffusivity=3e-7, half_life=1)
_Fdred = Molecule("Fd-red", 52, diffusivity=1e-7, half_life=1)
_hydroxybutyrylCoA = Molecule("hydroxybutyryl-CoA", 572, diffusivity=3e-7, half_life=1)
_succinylCoA = Molecule("succinyl-CoA", 445, diffusivity=3e-7)
_HSCoA = Molecule("HS-CoA", 1, diffusivity=3e-7, half_life=10)
_formylFH4 = Molecule("formyl-FH4", 57, diffusivity=3e-7, half_life=1)
_4hydroxybutyrate = Molecule("4-hydroxy-butyrate", 558, diffusivity=6e-6)
_malonylCoA = Molecule("malonyl-CoA", 252, diffusivity=3e-7, half_life=1)
_methylmalonylCoA = Molecule("methylmalonyl-CoA", 445, diffusivity=3e-7, half_life=1)
_malylCoA = Molecule("malyl-CoA", 445, diffusivity=3e-7, half_life=1)
_methylmalylCoA = Molecule("methylmalyl-CoA", 628, diffusivity=3e-7, half_life=1)
_CO = Molecule("CO", 125, diffusivity=2e-5, permeability=0.1)
_Ru5P = Molecule("Ru5P", 660, diffusivity=5e-6, half_life=100)
_13BPG = Molecule("1,3BPG", 360, diffusivity=4e-6, half_life=0.1)
_acetylCoA = Molecule("acetyl-CoA", 258, diffusivity=3e-7, half_life=1)
_Fered = Molecule("Fe-red", 63, diffusivity=1e-7, half_life=1)
_aketoglutarate = Molecule("a-ketoglutarate", 552, diffusivity=5e-6, half_life=100)
_NADPH = Molecule("NADPH", 68, diffusivity=2e-7, half_life=1)
_methylFH4 = Molecule("methyl-FH4", 153, diffusivity=3e-7, half_life=1)
_pyruvate = Molecule("pyruvate", 360, diffusivity=8e-6, half_life=30)
_CO2 = Molecule("CO2", 1, diffusivity=2e-5, permeability=0.1)
_FH4 = Molecule("FH4", 1, diffusivity=3e-7, half_life=1)
_methylenFH4 = Molecule("methylen-FH4", 109, diffusivity=3e-7, half_life=1)
_formate = Molecule("formate", 68, diffusivity=1e-5)
_succinicsemialdehyde = Molecule("succinic-semialdehyde", 516, diffusivity=6e-6, half_life=5)
_RuBP = Molecule("RuBP", 659, diffusivity=4e-6, half_life=10)
_NADP = Molecule("NADP", 1, diffusivity=2e-7)
_oxalacetate = Molecule("oxalacetate", 371, diffusivity=5e-6, half_life=1)
_FADH2 = Molecule("FADH2", 43, diffusivity=2e-7, half_life=1)
_citrate = Molecule("citrate", 618, diffusivity=5e-6)
_Feox = Molecule("Fe-ox", 1, diffusivity=1e-7, half_life=10)
_3PGA = Molecule("3PGA", 310, diffusivity=5e-6)
_FAD = Molecule("FAD", 1, diffusivity=2e-7)
_ADP = Molecule("ADP", 32, diffusivity=2e-6)
_G3P = Molecule("G3P", 396, diffusivity=5e-6, half_life=100)
_glyoxylate = Molecule("glyoxylate", 182, diffusivity=8e-6, half_life=30)
_succinate = Molecule("succinate", 424, diffusivity=6e-6)
_malate = Molecule("malate", 408, diffusivity=6e-6)
_PEP = Molecule("PEP", 397, diffusivity=5e-6, half_life=1)
_QH2 = Molecule("QH2", 3, diffusivity=1e-7, permeability=10, half_life=1)
_Fdox = Molecule("Fd-ox", 1, diffusivity=1e-7, half_life=10)
_ATP = Molecule("ATP", 62, diffusivity=2e-6, half_life=30)
_acetoacetylCoA = Molecule("acetoacetyl-CoA", 515, diffusivity=3e-7, half_life=0.1)
_propionylCoA = Molecule("propionyl-CoA", 451, diffusivity=3e-7, half_life=1)
_Pi = Molecule("Pi", 12, diffusivity=1e-5)
_PPi = Molecule("PPi", 13, diffusivity=5e-6)
_Q = Molecule("Q", 1, diffusivity=1e-7, permeability=10, half_life=10)

molecules = [
    _AMP,
    _fumarate,
    _isocitrate,
    _citramalylCoA,
    _Fdred,
    _hydroxybutyrylCoA,
    _succinylCoA,
    _HSCoA,
    _formylFH4,
    _4hydroxybutyrate,
    _malonylCoA,
    _methylmalonylCoA,
    _malylCoA,
    _methylmalylCoA,
    _CO,
    _Ru5P,
    _13BPG,
    _acetylCoA,
    _Fered,
    _aketoglutarate,
    _NADPH,
    _methylFH4,
    _pyruvate,
    _CO2,
    _FH4,
    _methylenFH4,
    _formate,
    _succinicsemialdehyde,
    _RuBP,
    _NADP,
    _oxalacetate,
    _FADH2,
    _citrate,
    _Feox,
    _3PGA,
    _FAD,
    _ADP,
    _G3P,
    _glyoxylate,
    _succinate,
    _malate,
    _PEP,
    _QH2,
    _Fdox,
    _ATP,
    _acetoacetylCoA,
    _propionylCoA,
    _Pi,
    _PPi,
    _Q,
]

reactions = [
    Reaction([_RuBP] + [_CO2], 2 * [_3PGA]),
    Reaction([_3PGA] + [_ATP], [_13BPG] + [_ADP]),
    Reaction([_13BPG] + [_NADPH], [_G3P] + [_NADP]),
    Reaction(5 * [_G3P], 3 * [_Ru5P]),
    Reaction([_Ru5P] + [_ATP], [_RuBP] + [_ADP]),
    Reaction([_acetylCoA] + [_CO2] + [_ATP], [_malonylCoA] + [_ADP] + [_Pi]),
    Reaction(
        [_malonylCoA] + 3 * [_NADPH] + [_ATP], [_propionylCoA] + 3 * [_NADP] + [_AMP] + [_PPi]
    ),
    Reaction([_propionylCoA] + [_CO2] + [_ATP], [_methylmalonylCoA] + [_ADP] + [_Pi]),
    Reaction([_methylmalonylCoA], [_succinylCoA]),
    Reaction([_succinylCoA], [_succinate] + [_HSCoA]),
    Reaction([_succinate] + [_Q], [_fumarate] + [_QH2]),
    Reaction([_fumarate], [_malate]),
    Reaction([_malate] + [_HSCoA], [_malylCoA]),
    Reaction([_malylCoA], [_acetylCoA] + [_glyoxylate]),
    Reaction([_propionylCoA] + [_glyoxylate], [_methylmalylCoA]),
    Reaction([_methylmalylCoA], [_citramalylCoA]),
    Reaction([_citramalylCoA], [_acetylCoA] + [_pyruvate]),
    Reaction([_CO2] + [_NADPH], [_formate] + [_NADP]),
    Reaction([_formate] + [_FH4] + [_ATP], [_formylFH4] + [_ADP] + [_Pi]),
    Reaction([_formylFH4] + [_NADPH], [_methylenFH4] + [_NADP]),
    Reaction([_methylenFH4] + [_NADPH], [_methylFH4] + [_NADP]),
    Reaction([_CO2] + 2 * [_Fered], [_CO] + 2 * [_Feox]),
    Reaction([_methylFH4] + [_CO] + [_HSCoA], [_acetylCoA] + [_FH4]),
    Reaction([_oxalacetate] + [_NADPH], [_malate] + [_NADP]),
    Reaction([_malate], [_fumarate]),
    Reaction([_fumarate] + [_FADH2], [_succinate] + [_FAD]),
    Reaction([_succinate] + [_HSCoA] + [_ATP], [_succinylCoA] + [_ADP] + [_Pi]),
    Reaction([_succinylCoA] + 2 * [_Fdred] + [_CO2], [_aketoglutarate] + [_HSCoA] + 2 * [_Fdox]),
    Reaction([_aketoglutarate] + [_CO2] + [_NADPH], [_isocitrate] + [_NADP]),
    Reaction([_isocitrate], [_citrate]),
    Reaction([_citrate] + [_HSCoA] + [_ATP], [_oxalacetate] + [_acetylCoA] + [_ADP] + [_Pi]),
    Reaction([_acetylCoA] + [_CO2] + 2 * [_Fdred], [_pyruvate] + 2 * [_Fdox] + [_HSCoA]),
    Reaction([_pyruvate] + [_ATP] + [_Pi], [_PEP] + [_AMP] + [_PPi]),
    Reaction([_PEP] + [_CO2], [_oxalacetate] + [_Pi]),
    Reaction([_oxalacetate] + [_NADPH], [_malate] + [_NADP]),
    Reaction([_malate], [_fumarate]),
    Reaction([_fumarate] + 2 * [_Fdred], [_succinate] + 2 * [_Fdox]),
    Reaction([_succinate] + [_HSCoA] + [_ATP], [_succinylCoA] + [_ADP] + [_Pi]),
    Reaction([_succinylCoA] + 2 * [_Fdred], [_succinicsemialdehyde] + [_HSCoA] + 2 * [_Fdox]),
    Reaction([_succinicsemialdehyde] + [_NADPH], [_4hydroxybutyrate] + [_NADP]),
    Reaction([_4hydroxybutyrate] + [_HSCoA] + [_ATP], [_hydroxybutyrylCoA] + [_AMP] + [_PPi]),
    Reaction([_hydroxybutyrylCoA] + [_NADP], [_acetoacetylCoA] + [_NADPH]),
    Reaction([_acetoacetylCoA] + [_HSCoA], 2 * [_acetylCoA]),
    Reaction([_acetylCoA] + [_CO2] + [_ATP], [_malonylCoA] + [_ADP] + [_Pi]),
    Reaction(
        [_malonylCoA] + 3 * [_NADPH] + [_ATP], [_propionylCoA] + 3 * [_NADP] + [_AMP] + [_PPi]
    ),
    Reaction([_propionylCoA] + [_CO2] + [_ATP], [_methylmalonylCoA] + [_ADP] + [_Pi]),
    Reaction([_methylmalonylCoA], [_succinylCoA]),
    Reaction([_succinylCoA] + [_NADPH], [_succinicsemialdehyde] + [_HSCoA] + [_NADP]),
    Reaction([_succinicsemialdehyde] + [_NADPH], [_4hydroxybutyrate] + [_NADP]),
    Reaction([_4hydroxybutyrate] + [_HSCoA] + [_ATP], [_hydroxybutyrylCoA] + [_AMP] + [_PPi]),
    Reaction([_hydroxybutyrylCoA] + [_NADP], [_acetoacetylCoA] + [_NADPH]),
    Reaction([_acetoacetylCoA] + [_HSCoA], 2 * [_acetylCoA]),
]

chemistry = Chemistry(molecules=molecules, reactions=reactions)
