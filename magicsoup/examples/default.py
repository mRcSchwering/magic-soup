from ..util import variants
from ..genetics import (
    Signal,
    Molecule,
    Action,
    DomainFact,
    ActionDomainFact,
    ReceptorDomainFact,
    SynthesisDomainFact,
)

MA = Molecule("A", 5)  # molceule A
MB = Molecule("B", 4)  # molceule B
MC = Molecule("C", 3)  # molceule C
MD = Molecule("D", 3)  # molceule D
ME = Molecule("E", 2)  # molceule E
MF = Molecule("F", 1)  # molceule F

MOLECULES: list[Signal] = [MA, MB, MC, MD, ME, MF]


CM = Action("CM", 2)  # cell migration
CR = Action("CR", 4)  # cell replication
DR = Action("DR", 1)  # DNA repair
TP = Action("TP", 1)  # transposon


ACTIONS: list[Signal] = [CM, CR, DR, TP]


# TODO: transporter domain: energy from concentration gradients
# TODO: calculate equilibrium constant from energy

DOMAINS: dict[DomainFact, list[str]] = {
    ActionDomainFact(CM): variants("GGNCNN") + variants("AGNTNN"),
    ActionDomainFact(CR): variants("GTNCNN") + variants("CTNGNN"),
    ActionDomainFact(DR): variants("CANANN") + variants("GGNTNN"),
    ActionDomainFact(TP): variants("CGNANN") + variants("GGNGNN"),
    ReceptorDomainFact(MA): variants("CTNTNN") + variants("CTNCNN"),
    ReceptorDomainFact(MB): variants("CGNCNN") + variants("AGNANN"),
    ReceptorDomainFact(MC): variants("AANCNN") + variants("GCNGNN"),
    ReceptorDomainFact(MD): variants("CANGNN") + variants("TTNANN"),
    ReceptorDomainFact(ME): variants("CANTNN") + variants("AANANN"),
    ReceptorDomainFact(MF): variants("CGNGNN") + variants("GANCNN"),
    SynthesisDomainFact(MA): variants("ATNCNN") + variants("CANCNN"),
    SynthesisDomainFact(MB): variants("ACNANN") + variants("CCNTNN"),
    SynthesisDomainFact(MC): variants("TCNCNN") + variants("CGNTNN"),
    SynthesisDomainFact(MD): variants("TGNTNN") + variants("ACNTNN"),
    SynthesisDomainFact(ME): variants("GGNANN") + variants("TANCNN"),
    SynthesisDomainFact(MF): variants("CTNANN") + variants("TANTNN"),
}
