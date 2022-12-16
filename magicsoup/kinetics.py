import torch
from .constants import EPS
from .containers import Protein, Molecule


def calc_cell_params(
    proteomes: list[list[Protein]],
    n_signals: int,
    mol_2_idx: dict[tuple[Molecule, bool], int],
    cell_idxs: list[int],
    Km: torch.Tensor,
    Vmax: torch.Tensor,
    E: torch.Tensor,
    N: torch.Tensor,
    A: torch.Tensor,
):
    """
    Generate cell-based parameter tensors from proteomes.
    Returns `(Km, Vmax, Ke, N, A)`:

    - `X` Signal/molecule concentrations
    - `Km` Domain affinities of all proteins for each signal
    - `Vmax` Maximum velocities of all proteins
    - `Ke` Equilibrium constants of all proteins. If the current reaction quotient of a protein
    exceeds its equilibrium constant, it will work in the other direction
    - `N` Reaction stoichiometry of all proteins for each signal. Numbers < 0.0 indicate this
    amount of this molecule is being used up by the protein. Numbers > 0.0 indicate this amount of this
    molecule is being created by the protein. 0.0 means this molecule is not part of the reaction of
    this protein.
    - `A` allosteric control of all proteins for each signal. Numbers > 0.0 mean this molecule acts as
    an activating effector on this protein, numbers < 0.0 mean this molecule acts as an inhibiting
    effector on this protein. 0.0 means this molecule does not allosterically effect the protein.
    """

    # TODO: maybe faster to first check which proteins in each cell actually changed,
    #       then only check (cell_i, protein_i) instead of all proteins for cell_i

    for cell_i, cell in zip(cell_idxs, proteomes):
        for prot_i, protein in enumerate(cell):
            energy = 0.0
            km: list[list[float]] = [[] for _ in range(n_signals)]
            vmax: list[float] = []
            a: list[int] = [0 for _ in range(n_signals)]
            n: list[int] = [0 for _ in range(n_signals)]

            for dom in protein.domains:

                if dom.is_allosteric:
                    mol = dom.substrates[0]
                    mol_i = mol_2_idx[mol, dom.is_transmembrane]
                    km[mol_i].append(dom.affinity)
                    a[mol_i] += -1 if dom.is_inhibiting else 1

                if dom.is_transporter:
                    vmax.append(dom.velocity)
                    mol = dom.substrates[0]

                    if dom.is_bkwd:
                        sub_i = mol_2_idx[mol, True]
                        prod_i = mol_2_idx[mol, False]
                    else:
                        sub_i = mol_2_idx[mol, False]
                        prod_i = mol_2_idx[mol, True]

                    km[sub_i].append(dom.affinity)
                    n[sub_i] -= 1

                    km[prod_i].append(1 / dom.affinity)
                    n[prod_i] += 1

                if dom.is_catalytic:
                    vmax.append(dom.velocity)

                    if dom.is_bkwd:
                        subs = dom.products
                        prods = dom.substrates
                    else:
                        subs = dom.substrates
                        prods = dom.products

                    for mol in subs:
                        energy -= mol.energy
                        mol_i = mol_2_idx[mol, False]
                        km[mol_i].append(dom.affinity)
                        n[mol_i] -= 1

                    for mol in prods:
                        energy += mol.energy
                        mol_i = mol_2_idx[mol, False]
                        km[mol_i].append(1 / dom.affinity)
                        n[mol_i] += 1

            E[cell_i, prot_i] = energy

            if len(vmax) > 0:
                Vmax[cell_i, prot_i] = sum(vmax) / len(vmax)

            for mol_i in range(n_signals):
                A[cell_i, prot_i, mol_i] = float(a[mol_i])
                N[cell_i, prot_i, mol_i] = float(n[mol_i])

                if len(km[mol_i]) > 0:
                    Km[cell_i, prot_i, mol_i] = sum(km[mol_i]) / len(km[mol_i])


# TODO: a molecule can be created by one domain and at the same time deconstructed
#       by another. Currently, this results in N = 0 for that molecule. With the current
#       implementation, the molecule will not be part of X_sub and X_prod, i.e. not part
#       of the MM equiation. Does this make sense???
#       Shouldn't that molecule then act more like an allosteric activator?


def integrate_signals(
    X: torch.Tensor,
    Km: torch.Tensor,
    Vmax: torch.Tensor,
    Ke: torch.Tensor,
    N: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate new molecules constructed or deconstructed by all proteins of all cells
    after 1 timestep. Returns the change in signals in shape `(c, s)`.
    There are `c` cells, `p` proteins, `s` signals/molecules.

    - `X` Signal/molecule concentrations (c, s). Must all be >= 0.0.
    - `Km` Domain affinities of all proteins for each molecule (c, p, s). Must all be >= 0.0.
    - `Vmax` Maximum velocities of all proteins (c, p). Must all be >= 0.0.
    - `Ke` Equilibrium constants of all proteins (c, p). If the current reaction quotient of a protein
    exceeds its equilibrium constant, it will work in the other direction. Must all be >= 0.0.
    - `N` Reaction stoichiometry of all proteins for each molecule (c, p, s). Numbers < 0.0 indicate this
    amount of this molecule is being used up by the protein. Numbers > 0.0 indicate this amount of this
    molecule is being created by the protein. 0.0 means this molecule is not part of the reaction of
    this protein.
    - `A` allosteric control of all proteins for each molecule (c, p, s). Numbers > 0.0 mean these molecules
    act as activating effectors, numbers < 0.0 mean these molecules act as inhibiting effectors.
    
    Everything is based on Michaelis Menten kinetics where protein velocity
    depends on substrate concentration:

    ```
        v = Vmax * x / (Km + x)
    ```

    `Vmax` is the maximum velocity of the protein and `Km` the substrate affinity. Multiple
    substrates create interaction terms such as:

    ```
        v = Vmax * x1 * / (Km1 + x1) * x2 / (Km2 + x2)
    ```

    Allosteric effectors work non-competitively such that they reduce or raise `Vmax` but
    leave any `Km` unaffected. Effectors themself also use Michaelis Menten kinteics.
    Here is substrate `x` with inhibitor `i`:
    
    ```
        v = Vmax * x / (Kmx + x) * (1 - Vi)
        Vi = i / (Kmi + i)
    ```

    Activating effectors effectively make the protein dependent on that activator:

    ```
        v = Vmax * x / (Kmx + x) * Va
        Va = a / (Kma + a)
    ```

    Multiple inhibitors and activators are each integrated with each other as if they
    were interacting with each other. I.e. they are multiplied when calculating their
    activity. Each term is clamped to `[0; 1]` before it is multiplied with `Vmax`.
    So a protein can never exceed `Vmax`.

    ```
        v = Vmax * x / (Km + x) * Va * (1 - Vi)
        Va = a1 * a2 / ((Kma1 + a1) * (Kma2 + a2))
        Vi = i1 * i2 / ((Kmi1 + i1) * (Kmi2 + i2))
    ```

    Equilibrium constants `Ke` define whether the reaction of a protein (as defined in `N`)
    can take place. If the reaction quotient (the combined concentrations of all products
    devided by all substrates) is smaller than its `Ke` the reaction proceeds. If it is
    greater than `Ke` the reverse reaction will take place (`N * -1.0` for this protein).
    The idea is to link signal integration to energy conversion.

    ```
        -dG0 = R * T * ln(Ke)
        Ke = exp(-dG0 / R / T)
    ```

    where `dG0` is the standard Gibb's free energy of this reaction, `R` is gas constant,
    `T` is absolute temperature.

    While the reaction quotient `Q` is well below _Ke_ the protein will proceed with the reaction
    in forward direction with a velocity as defined by the above equiations in each time step.
    However, as `Q` approaches `Ke` there is a chance that in one time step `Q` surpasses `Ke`.
    In the next time step the reaction would turn around and `Q` might possibly jump again below
    `Ke`. Thus, in a sensitive protein (low `Km`, high `Vmax`, no inhibition) `Q` might jump
    above and below `Ke` with every time step, i.e. always overshootig the equilibrium state.
    
    One would have to solve for `v` given `Ke` and the stoichiometry in order to limit `v`
    to reach the exact equilibrium state. The equiation has this form:

    ```
        ([P] + m * v)^m / ([S] + n * v)^n = Ke
    ```

    Where `[P]` and `[S]` are product and substrate concentrations and `m` and `n` are their
    stoichiometric coefficients (extends as product with multiple substrate and product species).

    I can't solve this in general. Thus, to limit the extend of such a flickering around the
    equilibrium state `Km`s of the backwards reaction should the inverse of the forward reaction.
    E.g. if the forward reaction was very sensitive and tends to overshoot the equilibrium state,
    the backward reaction will be very unsensitive and only slowly approach the equilibrium state
    from the other side. As transporters also work with this mechanism, they tend to have a
    direction. So, if a transporter has a high affinity in one direction, it will have a low
    affinity in the other.

    Somehwat related: As reactions can overshoot their equilibriums state, they can also
    overshoot the actual substrate concentrations. I.e. it can happen that a protein tries to
    react with more substrate than actually available (e.g. very low `Km`, high `Vmax`). In
    addition, each protein in the cell is not aware of the other proteins. So, 2 proteins could
    both try to react with substrate `S`. And while each of them would have had enough `S`, both
    of them together actually reduce `S` to a negative concentration.

    To avoid this there is a correction term that limits the velocity of any protein to the actual
    abundance of the substrates. If there are multiple proteins in a cell all using up the same
    substrate, these limiting substrate abundances are shared equally among these proteins.
    Furthermore, instead of allowing the proteins to reduce substrates to exactly `0.0`, they will
    actually only be allowed to reduce any substrate to a small number `eps > 0.0`. This is to
    avoid creating infinite reaction quotients and it is another mitigation for the problem of
    overshooting the equilibirum state.

    Limitations:
    - all based on Michaelis-Menten kinetics, no cooperativity
    - all allosteric control is non-competitive (activating or inhibiting)
    - there are substrate-substrate interactions but no interactions among effectors
    - 1 protein can have multiple substrates and products but there is only 1 Km for each type of molecule
    - there can only be 1 effector per molecule (e.g. not 2 different allosteric controls for the same type of molecule)
    - proteins can catalyze reactions in a way that they overshoot their equilibirum state (heuristics try to avoid that, see text above)
    """
    # TODO: refactor:
    #       - it seems like some of the masks are unnecessary (e.g. if I know that in
    #         N there is 0.0 in certain places, do I really need a mask like sub_M?)
    #       - can I first do the nom / denom, then pow and prod afterwards?
    #       - are there some things which I basically calculate 2 times? Can I avoid it?
    #         or are there some intermediates which could be reused if I would calculate
    #         them slightly different/in different order?
    #       - does it help to use masked tensors? (pytorch.org/docs/stable/masked.html)
    #       - better names, split into a few functions?
    #       - torch.expand faster?

    # substrates
    sub_M = torch.where(N < 0.0, 1.0, 0.0)  # (c, p, s)
    sub_X = torch.einsum("cps,cs->cps", sub_M, X)  # (c, p, s)
    sub_N = torch.where(N < 0.0, -N, 0.0)  # (c, p, s)

    # products
    pro_M = torch.where(N > 0.0, 1.0, 0.0)  # (c, p s)
    pro_X = torch.einsum("cps,cs->cps", pro_M, X)  # (c, p, s)
    pro_N = torch.where(N > 0.0, N, 0.0)  # (c, p, s)

    # quotients
    nom = torch.pow(pro_X, pro_N).prod(2)  # (c, p)
    denom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
    Q = nom / denom  # (c, p)

    # adjust direction
    adj_N = torch.where(Q > Ke, -1.0, 1.0)  # (c, p)
    N_adj = torch.einsum("cps,cp->cps", N, adj_N)

    # inhibitors
    inh_N = torch.where(A < 0.0, -A, 0.0)  # (c, p, s)
    inh_M = torch.any(A < 0.0, dim=2)  # (c, p)
    inh_X = torch.einsum("cps,cs->cps", inh_N, X)  # (c, p, s)
    nom = torch.pow(inh_X, inh_N).prod(2)  # (c, p)
    denom = torch.pow(Km + inh_X, inh_N).prod(2)  # (c, p)
    inh_V = torch.where(inh_M, nom / denom, 0.0).clamp(0.0, 1.0)  # (c, p)

    # activators
    act_N = torch.where(A > 0.0, A, 0.0)  # (c, p, s)
    act_M = torch.any(A > 0.0, dim=2)  # (c, p)
    act_X = torch.einsum("cps,cs->cps", act_N, X)  # (c, p, s)
    nom = torch.pow(act_X, act_N).prod(2)  # (c, p)
    denom = torch.pow(Km + act_X, act_N).prod(2)  # (c, p)
    act_V = torch.where(act_M, nom / denom, 1.0).clamp(0.0, 1.0)  # (c, p)

    # substrates
    sub_M = torch.where(N_adj < 0.0, 1.0, 0.0)  # (c, p s)
    sub_X = torch.einsum("cps,cs->cps", sub_M, X)  # (c, p, s)
    sub_N = torch.where(N_adj < 0.0, -N_adj, 0.0)  # (c, p, s)

    # proteins
    prot_Vmax = sub_M.max(dim=2).values * Vmax  # (c, p)
    nom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
    denom = torch.pow(Km + sub_X, sub_N).prod(2)  # (c, p)
    prot_V = prot_Vmax * nom / denom * (1 - inh_V) * act_V  # (c, p)

    # concentration deltas (c, s)
    Xd = torch.einsum("cps,cp->cs", N_adj, prot_V)

    X1 = X + Xd
    if torch.any(X1 < EPS):
        neg_conc = torch.where(X1 < EPS, 1.0, 0.0)  # (c, s)
        candidates = torch.where(N_adj < 0.0, 1.0, 0.0)  # (c, p, s)

        # which proteins need to be down-regulated (c, p)
        BX_mask = torch.einsum("cps,cs->cps", candidates, neg_conc)
        prot_M = BX_mask.max(dim=2).values

        # what are the correction factors (c,)
        correct = torch.where(neg_conc > 0.0, -(X - EPS) / Xd, 1.0).min(dim=1).values

        # correction for protein velocities (c, p)
        prot_V_adj = torch.einsum("cp,c->cp", prot_M, correct)
        prot_V_adj = torch.where(prot_V_adj == 0.0, 1.0, prot_V_adj)

        # new concentration deltas (c, s)
        Xd = torch.einsum("cps,cp->cs", N_adj, prot_V * prot_V_adj)

        # TODO: can I already predict this before calculated Xd the first time?
        #       maybe at the time when I know the intended protein velocities?
        #       and then use this as the maximum Vmax, so that protein velocities
        #       are capped automatically capped to this velocity if substrate
        #       concentrations are low

    return Xd
