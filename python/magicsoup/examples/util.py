from collections import Counter
from collections.abc import Iterable

import z3


class Reaction:
    def __init__(self, descr: str) -> None:
        assert " -> " in descr
        lft, rgt = descr.split(" -> ")
        rgt, rest = rgt.split("(")
        rest = rest.split(")")[0]
        assert "kJ/mol" in rest
        rgt = rgt.strip()
        lft = lft.strip()
        rest = rest.strip()

        self.energy = int(rest.split("kJ/mol")[0].strip())
        self.substrates = self._extract(lft)
        self.products = self._extract(rgt)
        self.molecules = set(self.substrates) | set(self.products)
        self.m2s = {d: "_" + d.replace(",", "").replace("-", "") for d in self.molecules}

    def _extract(self, side: str) -> list[str]:
        summands = side.split("+")
        out = []
        for summand in summands:
            parts = [d for d in summand.strip().split() if len(d) > 0]
            assert len(parts) < 3
            if len(parts) == 2:
                out.extend([parts[1].strip()] * int(parts[0]))
            else:
                out.append(parts[0].strip())
        return out


class ChemistrySolver:
    def __init__(
        self, reacts: list[Reaction], other_conds: Iterable[str] = (), abserr: int = 15
    ) -> None:
        self.abserr = abserr
        self.m2v: dict[str, str] = {}
        self.molecules: set[str] = set()
        self.reactions = reacts
        for react in reacts:
            for mol in react.molecules:
                self.molecules.add(mol)
                slug = react.m2s[mol]
                if mol in self.m2v:
                    assert self.m2v[mol] == slug
                else:
                    self.m2v[mol] = slug

        self.init_conds = [f"{self.m2v[d]} > 0" for d in self.molecules]
        self.other_conds = [
            " ".join([self.m2v.get(dd, dd) for dd in d.split()]) for d in other_conds
        ]

    def solve(self) -> dict[str, int]:
        """Solve by minimizing for absolute error

        Defines all variables as greater 0 and solves for minimizing
        the absolute difference between expected reaction energy and actual reaction energy
        min(|exp_energy - act_energy|) while not allowing the absolute difference to be
        greater than the absolute acceptable error per reaction.
        """
        opt = z3.Optimize()

        for mol in self.molecules:
            exec(f"{self.m2v[mol]} = z3.Int('{mol}')")
            opt.add(eval(f"{self.m2v[mol]} > 0"))

        for cond in self.init_conds + self.other_conds:
            opt.add(eval(cond))

        errors = []
        for i, react in enumerate(self.reactions):
            subs = [eval(self.m2v[d]) for d in react.substrates]
            prods = [eval(self.m2v[d]) for d in react.products]
            d_i = sum(prods) - sum(subs)
            target = react.energy

            e_i = z3.Int(f"e_{i}")
            opt.add(e_i >= d_i - target)  # e_i >= |deviation| (upper side)
            opt.add(e_i >= target - d_i)  # e_i >= |deviation| (lower side)
            opt.add(e_i <= self.abserr)
            errors.append(e_i)

        total_error = z3.Sum(errors)
        opt.minimize(total_error)
        assert str(opt.check()) == "sat"
        res = opt.model()
        return {m: res[eval(self.m2v[m])] for m in self.molecules}


t = """RuBP + CO2 -> 2 3PGA (-40 kJ/mol)
3PGA + ATP -> 1,3BPG + ADP (20 kJ/mol)
1,3BPG + NADPH -> G3P + NADP (-31 kJ/mol)
5 G3P -> 3 Ru5P (0 kJ/mol)
Ru5P + ATP -> RuBP + ADP (-25 kJ/mol)
acetyl-CoA + CO2 + ATP -> malonyl-CoA + ADP + Pi (-25 kJ/mol)
malonyl-CoA + 3 NADPH + ATP -> propionyl-CoA + 3 NADP + AMP + PPi (-50 kJ/mol)
propionyl-CoA + CO2 + ATP -> methylmalonyl-CoA + ADP + Pi (-25 kJ/mol)
methylmalonyl-CoA -> succinyl-CoA (0 kJ/mol)
succinyl-CoA -> succinate + HS-CoA (-35 kJ/mol)
succinate + Q -> fumarate + QH2 (-10 kJ/mol)
fumarate -> malate (-4 kJ/mol)
malate + HS-CoA -> malyl-CoA (35 kJ/mol)
malyl-CoA -> acetyl-CoA + glyoxylate (-5 kJ/mol)
propionyl-CoA + glyoxylate -> methylmalyl-CoA (-5 kJ/mol)
methylmalyl-CoA -> citramalyl-CoA (0 kJ/mol)
citramalyl-CoA -> acetyl-CoA + pyruvate (-10 kJ/mol)
CO2 + NADPH -> formate + NADP (0 kJ/mol)
formate + FH4 + ATP -> formyl-FH4 + ADP + Pi (-30 kJ/mol)
formyl-FH4 + NADPH -> methylen-FH4 + NADP (-15 kJ/mol)
methylen-FH4 + NADPH -> methyl-FH4 + NADP (-23 kJ/mol)
CO2 + 2 Fe-red -> CO + 2 Fe-ox (0 kJ/mol)
methyl-FH4 + CO + HS-CoA -> acetyl-CoA + FH4 (-20 kJ/mol)
oxalacetate + NADPH -> malate + NADP (-30 kJ/mol)
malate -> fumarate (4 kJ/mol)
fumarate + FADH2 -> succinate + FAD (-30 kJ/mol)
succinate + HS-CoA + ATP -> succinyl-CoA + ADP + Pi (2 kJ/mol)
succinyl-CoA + 2 Fd-red + CO2 -> a-ketoglutarate + HS-CoA + 2 Fd-ox (5 kJ/mol)
a-ketoglutarate + CO2 + NADPH -> isocitrate + NADP (+12 kJ/mol)
isocitrate -> citrate (-14 kJ/mol)
citrate + HS-CoA + ATP -> oxalacetate + acetyl-CoA + ADP + Pi (-8 kJ/mol)
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
acetyl-CoA + CO2 + ATP -> malonyl-CoA + ADP + Pi (-25 kJ/mol)
malonyl-CoA + 3 NADPH + ATP -> propionyl-CoA + 3 NADP + AMP + PPi (-50 kJ/mol)
propionyl-CoA + CO2 + ATP -> methylmalonyl-CoA + ADP + Pi (-25 kJ/mol)
methylmalonyl-CoA -> succinyl-CoA (0 kJ/mol)
succinyl-CoA + NADPH -> succinic-semialdehyde + HS-CoA + NADP (5 kJ/mol)
succinic-semialdehyde + NADPH -> 4-hydroxy-butyrate + NADP (-25 kJ/mol)
4-hydroxy-butyrate + HS-CoA + ATP -> hydroxybutyryl-CoA + AMP + PPi (-35 kJ/mol)
hydroxybutyryl-CoA + NADP -> acetoacetyl-CoA + NADPH (10 kJ/mol)
acetoacetyl-CoA + HS-CoA -> 2 acetyl-CoA (0 kJ/mol)"""

others = """ATP > ADP, ADP > AMP, PPi > Pi
NADPH > NADP, FADH2 > FAD, Fe-red > Fe-ox, Fd-red > Fd-ox, QH2 > Q
acetyl-CoA > HS-CoA, Ru5P > RuBP, citramalyl-CoA > acetyl-CoA, methyl-FH4 > FH4, acetoacetyl-CoA > acetyl-CoA
ATP - ADP >= 30, NADPH - NADP >= 40, Fd-red - Fd-ox >= 40, acetyl-CoA - HS-CoA >= 30"""

solver = ChemistrySolver(
    reacts=[Reaction(d) for d in t.split("\n")],
    other_conds=[dd.strip() for d in others.split("\n") for dd in d.split(",")],
)
a = solver.solve()
for m, e in a.items():
    print(f'{solver.m2v[m]} = Molecule("{m}", {e})')

", ".join([solver.m2v[d] for d in a])


for react in solver.reactions:
    subs_cntr = Counter([solver.m2v[d] for d in react.substrates])
    prods_cntr = Counter([solver.m2v[d] for d in react.products])
    subs = " + ".join([f"{n} * [{d}]" if n > 1 else f"[{d}]" for d, n in subs_cntr.items()])
    prods = " + ".join([f"{n} * [{d}]" if n > 1 else f"[{d}]" for d, n in prods_cntr.items()])
    print(f"Reaction({subs}, {prods}),")
