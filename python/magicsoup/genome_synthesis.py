from typing import Protocol

from magicsoup.chemistry import Molecule, Reaction
from magicsoup.world import World

# TODO: why do I need these to_dict methods?


class DomainFactType(Protocol):
    """Protocol for domain factories"""

    def validate(self, world: World): ...

    def gen_coding_sequence(self, world: World) -> str: ...

    @classmethod
    def from_dict(cls, dct: dict) -> "DomainFactType": ...


class CatalyticDomainFact:
    """
    Factory for generating nucleotide sequences with [CatalyticDomains][magicsoup.containers.CatalyticDomain].

    Arguments:
        reaction: Tuple of `(substrates, products)` from [Chemistry][magicsoup.containers.Chemistry]
            where both `substrates` and `products` are lists of [Molecules][magicsoup.containers.Molecule].

    For stoichiometric coefficients > 1, list the molecule species multiple times.
    E.g. for `2A + B <-> C` use `reaction=([A, A, B], [C])`
    when `A,B,C` are molecule A, B, C instances.
    """

    def __init__(self, reaction: Reaction):
        self.reaction = reaction

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        if self.reaction not in world.chemistry.reactions:
            raise ValueError(
                f"CatalyticDomainFact has this reaction defined: {self.reaction}."
                " But this world's chemistry doesn't define this reaction."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        return world.cells.genomics.get_catalytic_domain_seq(
            reaction=self.reaction,
            react_2_tokens=world.cells.proteomics.reaction_map.react_2_tokens,
        )

    @classmethod
    def from_dict(cls, dct: dict, molmap: dict[str, Molecule]) -> "CatalyticDomainFact":
        """
        Generate domain factory from dct representation of a domain.
        A dct representation of this domain is returned by
        [CatalyticDomain.to_dict()][magicsoup.containers.CatalyticDomain.to_dict].
        """
        dct = dct["spec"]
        subs, prods = dct["reaction"]
        reaction = Reaction(
            substrates=[molmap[d] for d in subs],
            products=[molmap[d] for d in prods],
        )
        return cls(reaction=reaction)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.reaction})"

    def __str__(self) -> str:
        return str(self.reaction)
