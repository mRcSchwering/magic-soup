import torch
import random
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


def _get_kinetics() -> ms.Kinetics:
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=REACTIONS)
    kinetics = ms.Kinetics(
        molecules=chemistry.molecules,
        reactions=chemistry.reactions,
        abs_temp=310,
        scalar_enc_size=61,
        vector_enc_size=3904,
    )
    return kinetics


def test_cell_params_are_always_set_reproduceably():
    n_cells = 100
    n_tries = 10

    for i in range(n_tries):
        proteomes = []
        for _ in range(n_cells):
            proteins = []
            for _ in range(random.choice(list(range(20)))):
                domains = []
                for _ in range(random.choice([1, 1, 2])):
                    domtype = random.choice([1, 2, 3])
                    i0 = random.choice(list(range(61)))
                    i1 = random.choice(list(range(61)))
                    i2 = random.choice(list(range(61)))
                    i3 = random.choice(list(range(3904)))
                    domains.append(((domtype, i0, i1, i2, i3), 1, 2))
                proteins.append(domains)
            proteomes.append(proteins)

        n_max_prots = max(len(d) for d in proteomes)
        idxs = list(range(n_cells))

        kinetics = _get_kinetics()
        kinetics.increase_max_cells(by_n=n_cells)
        kinetics.increase_max_proteins(max_n=n_max_prots)
        kinetics.set_cell_params(cell_idxs=idxs, proteomes=proteomes)

        N_orig = kinetics.N.clone()
        Nb_orig = kinetics.Nb.clone()
        Nf_orig = kinetics.Nf.clone()
        Ke_orig = kinetics.Ke.clone()
        Kmf_orig = kinetics.Kmf.clone()
        Kmb_orig = kinetics.Kmb.clone()
        Kmr_orig = kinetics.Kmr.clone()
        Vmax_orig = kinetics.Vmax.clone()

        kinetics.remove_cell_params(keep=torch.full((n_cells,), False))
        kinetics.increase_max_cells(by_n=n_cells)
        kinetics.increase_max_proteins(max_n=n_max_prots)
        kinetics.set_cell_params(cell_idxs=idxs, proteomes=proteomes)

        assert torch.equal(kinetics.N, N_orig), i
        assert torch.equal(kinetics.Nb, Nb_orig), i
        assert torch.equal(kinetics.Nf, Nf_orig), i
        assert torch.equal(kinetics.Ke, Ke_orig), i
        assert torch.equal(kinetics.Kmf, Kmf_orig), i
        assert torch.equal(kinetics.Kmb, Kmb_orig), i
        assert torch.equal(kinetics.Kmr, Kmr_orig), i
        assert torch.equal(kinetics.Vmax, Vmax_orig), i


def test_random_kinetics_stay_zero():
    # Typical reasons for cells creating signals from 0:
    # 1. when using exp(ln(x)) 1s can accidentally be created
    # 2. when doing a^b 1s can be created (0^1=0 but 0^0=1)
    n_cells = 100
    n_prots = 100
    n_steps = 1000

    kinetics = _get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = torch.zeros(n_cells, n_mols).abs()

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-8, high=9, size=(n_cells, n_prots, n_mols)).float()
    kinetics.Nf = torch.where(kinetics.N < 0.0, -kinetics.N, 0.0)
    kinetics.Nb = torch.where(kinetics.N > 0.0, kinetics.N, 0.0)

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs() * 100

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-5, high=5, size=(n_cells, n_prots, n_mols))

    # affinities (c, p)
    Ke = torch.randn(n_cells, n_prots) * 100
    kinetics.Kmf = torch.randn(n_cells, n_prots).abs()
    kinetics.Kmb = kinetics.Kmf * Ke
    kinetics.Kmr = torch.randn(n_cells, n_prots, n_mols).abs()
    kinetics.Ke = kinetics.Kmb / kinetics.Kmf

    # test
    for _ in range(n_steps):
        X = kinetics.integrate_signals(X=X)
        assert X.min() == 0.0
        assert X.max() == 0.0


def test_random_kinetics_dont_explode():
    # interaction terms can overflow float32 when they become too big
    # i.g. exponents too high and many substrates
    # this will create infinites which will eventually become NaN
    n_cells = 100
    n_prots = 100
    n_steps = 1000

    kinetics = _get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = torch.randn(n_cells, n_mols).abs().clamp(max=1.0) * 100

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-8, high=9, size=(n_cells, n_prots, n_mols)).float()
    kinetics.Nf = torch.where(kinetics.N < 0.0, -kinetics.N, 0.0)
    kinetics.Nb = torch.where(kinetics.N > 0.0, kinetics.N, 0.0)

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs().clamp(max=1.0) * 100

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-5, high=5, size=(n_cells, n_prots, n_mols))

    # affinities (c, p)
    Ke = torch.randn(n_cells, n_prots) * 100
    kinetics.Kmf = torch.randn(n_cells, n_prots).abs().clamp(0.001)
    kinetics.Kmb = (kinetics.Kmf * Ke).clamp(0.001)
    kinetics.Kmr = torch.randn(n_cells, n_prots, n_mols).abs().clamp(0.001)
    kinetics.Ke = kinetics.Kmb / kinetics.Kmf

    # test
    for _ in range(n_steps):
        X = kinetics.integrate_signals(X=X)
        assert not torch.any(X < 0.0), X[X < 0.0].min()
        assert not torch.any(X.isnan()), X.isnan().sum()
        assert torch.all(X.isfinite()), ~X.isfinite().sum()
        assert torch.all(X < 10_000), X[X >= 10_000]
