from typing import Any


class Signal:
    def __init__(self, name: str, energy: float, is_intracellular=True):
        self.name = name
        self.energy = energy
        self.is_intracellular = is_intracellular

    def __hash__(self) -> int:
        clsname = type(self).__name__
        return hash((clsname, self.name, self.energy, self.is_intracellular))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(name=%r,energy=%r,is_intracellular=%r)" % (
            clsname,
            self.name,
            self.energy,
            self.is_intracellular,
        )

    def __str__(self) -> str:
        prefix = "i" if self.is_intracellular else "e"
        return prefix + "-" + self.name


class Molecule(Signal):
    pass


class Action(Signal):
    pass


class Domain:
    def __init__(
        self,
        in_sigs: tuple[Signal, ...],
        out_sigs: tuple[Signal, ...],
        affinity: float,
        velocity: float,
        energy: float,
        is_transporter=False,
        is_receptor=False,
        is_action=False,
    ):
        self.in_sigs = in_sigs
        self.out_sigs = out_sigs
        self.affinity = affinity
        self.velocity = velocity
        self.energy = energy

        self.is_transporter = is_transporter
        self.is_receptor = is_receptor
        self.is_action = is_action

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(in_sigs=%r,out_sigs=%r,affinity=%r,velocity=%r,is_transporter=%r,is_receptor=%r,is_action=%r)"
            % (
                clsname,
                self.in_sigs,
                self.out_sigs,
                self.affinity,
                self.velocity,
                self.is_transporter,
                self.is_receptor,
                self.is_action,
            )
        )

    def __str__(self) -> str:
        ins = ",".join(str(d) for d in self.in_sigs)
        outs = ",".join(str(d) for d in self.out_sigs)
        if self.is_transporter:
            return f"TransporterDomain({ins}->{outs})"
        if self.is_receptor:
            return f"ReceptorDomain({ins})"
        if self.is_action:
            return f"ActionDomain({outs})"
        return f"EnzymeDomain({ins}->{outs})"


class EnzymeFact:
    def __init__(
        self, reactions: list[tuple[tuple[Molecule, ...], tuple[Molecule, ...]]],
    ):
        self.reactions = reactions

        energies: list[float] = []
        for substrates, products in reactions:
            energy = 0.0
            for sig in substrates:
                energy -= sig.energy
            for sig in products:
                energy += sig.energy
            energies.append(energy)

        self.energies = energies

    def __call__(self, reaction_idx: int, affinity: float, velocity: float) -> Domain:
        subs, prods = self.reactions[reaction_idx]
        return Domain(
            in_sigs=subs,
            out_sigs=prods,
            affinity=affinity,
            velocity=velocity,
            energy=self.energies[reaction_idx],
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(reactions=%r)" % (clsname, self.reactions)


class TransporterFact:
    def __init__(self, molecules: list[Molecule]):
        self.molecules = molecules
        self.energy = 0.0

    def __call__(self, molecule_idx: int, affinity: float, velocity: float) -> Domain:
        mol1 = self.molecules[molecule_idx]
        mol2 = Molecule(
            name=mol1.name,
            energy=mol1.energy,
            is_intracellular=not mol1.is_intracellular,
        )
        return Domain(
            in_sigs=(mol1,),
            out_sigs=(mol2,),
            affinity=affinity,
            velocity=velocity,
            energy=self.energy,
            is_transporter=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r)" % (clsname, self.molecules)


class ReceptorFact:
    def __init__(self, molecules: list[Molecule]):
        self.molecules = molecules
        self.energy = 0.0

    def __call__(self, molecule_idx: int, affinity: float) -> Domain:
        return Domain(
            in_sigs=(self.molecules[molecule_idx],),
            out_sigs=tuple(),
            affinity=affinity,
            velocity=0.0,
            energy=self.energy,
            is_receptor=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r)" % (clsname, self.molecules)


class ActionFact:
    def __init__(self, actions: list[Action]):
        self.actions = actions

    def __call__(self, action_idx: int, affinity: float, velocity: float) -> Domain:
        act = self.actions[action_idx]
        return Domain(
            in_sigs=tuple(),
            out_sigs=(act,),
            affinity=affinity,
            velocity=velocity,
            energy=act.energy,
            is_action=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(actions=%r)" % (clsname, self.actions)


##
a = Molecule("a", 10.0)
b = Molecule("b", 8.0)
c = Molecule("c", 6.0)
d = Molecule("d", 4.0)
e = Molecule("e", 2.0)

f = Action("f", 10.0)
g = Action("g", 10.0)

molecules = [a, b, c, d, e]
actions = [f, g]

reactions: Any = [
    ((a,), (b,)),
    ((a, b), (c,)),
    ((c), (d,)),
    ((c, d), (e,)),
]

EnzymeDomain = EnzymeFact(reactions=reactions)
TransporterDomain = TransporterFact(molecules=molecules)
ReceptorDomain = ReceptorFact(molecules=molecules)
ActionDomain = ActionFact(actions=actions)

