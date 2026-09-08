from functools import reduce

import numpy as np
import numpy.typing as npt
import pytest

from vsa import *

# Fixtures


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=67)


@pytest.fixture(autouse=True)
def rest_rhc_state():
    saved = (RHC.moduli, RHC.basis, RHC.basis_exponents, RHC.anti_basis)
    yield
    RHC.moduli, RHC.basis, RHC.basis_exponents, RHC.anti_basis = saved


def make_codebooks[T: np.generic](
    vsa: type[VSA[T]], rng: np.random.Generator, num_factors: int, size: int, dim: int
) -> list[npt.NDArray[T]]:
    return [
        np.stack([vsa.new(dim).data for _ in range(size)]) for _ in range(num_factors)
    ]


def compose[T: np.generic](
    vsa: type[VSA[T]], codebooks: list[npt.NDArray[T]], indices: tuple[int, ...]
) -> npt.NDArray[T]:
    return reduce(vsa.bind, [cb[k] for cb, k in zip(codebooks, indices, strict=True)])


# Nonlinearities


def test_phasor_projection_yields_unit_modulus(rng: np.random.Generator):
    x = rng.normal(size=(100,)) + 1j * rng.normal(size=(100,))
    assert np.allclose(np.abs(phasor_projection(x)), 1.0)


def test_phasor_projection_preserves_phase(rng: np.random.Generator):
    x = rng.normal(size=(100,)) + 1j * rng.normal(size=(100,))
    assert np.allclose(np.angle(phasor_projection(x)), np.angle(x))


def test_phasor_projection_is_idempotent(rng: np.random.Generator):
    u = FHRR.new(1000).data
    assert np.allclose(phasor_projection(u), u)


def test_phasor_projection_discards_magnitude(rng: np.random.Generator):
    x = rng.normal(size=(100,)) + 1j * rng.normal(size=(100,))
    assert np.allclose(phasor_projection(x), phasor_projection(x * 18.0))


def test_phasor_projection_handles_zero(rng: np.random.Generator):
    x = np.array([0j, 1 + 0j])
    assert np.isfinite(phasor_projection(x)).all()


# Resonator networks


def test_projection_recovers_one_hot_coefficients(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, 1, size=10, dim=2000)
    cb = cbs[0]
    net = ResonatorNetwork(FHRR, cbs, non_linearity=phasor_projection)
    coeffs = net.project(cb, cb[3])
    assert np.argmax(np.abs(coeffs)) == 3
    assert np.abs(coeffs[3]) == pytest.approx(1.0, abs=0.05)
    assert np.abs(np.delete(coeffs, 3)).max() < 0.2


def test_cleanup_i_identity_on_codebook_entry(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, 1, size=10, dim=2000)
    cb = cbs[0]
    net = ResonatorNetwork(FHRR, cbs, non_linearity=phasor_projection)
    est, _ = net.cleanup(cb, cb[3])
    assert FHRR.similarity(est, cb[3]) > 0.95


def test_residual_recovers_the_held_out_factor(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=2000)
    net = ResonatorNetwork(FHRR, cbs, non_linearity=phasor_projection)
    s = compose(FHRR, cbs, (1, 2, 3))
    r = net.residual(s, [cb[k] for cb, k in zip(cbs, (1, 2, 3), strict=True)], i=1)
    assert FHRR.similarity(r, cbs[1][2]) > 0.95


def test_correct_factorization_is_fixpoint(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=2000)
    net = ResonatorNetwork(FHRR, cbs, non_linearity=phasor_projection)
    indices = (1, 2, 3)
    truth = [cb[k] for cb, k in zip(cbs, indices, strict=True)]
    s = compose(FHRR, cbs, indices)
    new, _ = net.step(s, truth)
    for a, b in zip(new, truth, strict=True):
        assert FHRR.similarity(a, b) > 0.95


def test_step_output_stays_in_the_state_space(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=2000)
    net = ResonatorNetwork(FHRR, cbs, non_linearity=phasor_projection)
    indices = (1, 2, 3)
    truth = [cb[k] for cb, k in zip(cbs, indices, strict=True)]
    s = compose(FHRR, cbs, indices)
    new, _ = net.step(s, truth)
    assert all(np.allclose(np.abs(e), 1.0) for e in new)


def test_initial_estimates_shape_and_count(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=2000)
    net = ResonatorNetwork(FHRR, cbs, non_linearity=phasor_projection)
    ests = net.initial_estimates()
    assert len(ests) == net.num_factors
    assert all(e.shape == (net.dim,) for e in ests)


def test_dim_is_vec_dim_not_codebook_size(rng: np.random.Generator):
    net = ResonatorNetwork(
        vsa=FHRR, codebooks=make_codebooks(FHRR, rng, num_factors=3, size=7, dim=512)
    )
    assert net.dim == 512 and net.num_factors == 3


def test_misimatched_dimensions_rejected():
    with pytest.raises(AssertionError):
        ResonatorNetwork(vsa=FHRR, codebooks=[np.zeros((5, 512)), np.zeros((5, 256))])


@pytest.mark.parametrize("indices", [(0, 0, 0), (3, 7, 1), (9, 9, 9)])
def test_decode_recovers_known_factors(
    rng: np.random.Generator, indices: tuple[int, int, int]
):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=1500)
    net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
    result = net.decode(compose(FHRR, cbs, indices))
    assert result.indices == indices
    assert result.similarity > net.sim_threshold
    assert result.iters < 50


def test_synchronous_and_sequential_both_converge(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=1500)
    net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
    indices = (1, 2, 3)
    s = compose(FHRR, cbs, indices)
    for synchronous in [True, False]:
        assert net.with_(synchronous=synchronous).decode(s).indices == indices


def test_decode_is_deterministic(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=1500)
    net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
    indices = (1, 2, 3)
    s = compose(FHRR, cbs, indices)
    assert net.decode(s).indices == net.decode(s).indices


# def test_single_factor_degenerates_to_cleanup(rng: np.random.Generator):
#     cbs = make_codebooks(FHRR, rng, num_factors=1, size=10, dim=1500)
#     cb = cbs[0]
#     net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
#     assert net.decode(cb[4]).indices == (4,)


def test_raises_on_target_that_is_not_product(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=1500)
    net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
    with pytest.raises(ConvergenceError):
        net.decode(FHRR.new(1500).data)


def test_raises_when_over_capacity(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=50, dim=10)
    small_net = ResonatorNetwork(
        vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection
    )
    indices = tuple(x + 1 for x in range(3))
    s = compose(FHRR, cbs, indices)
    with pytest.raises(ConvergenceError):
        small_net.decode(s)


def test_raises_when_iteration_budget_exhausted(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=1500)
    net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
    indices = (1, 2, 3)
    s = compose(FHRR, cbs, indices)
    with pytest.raises(ConvergenceError):
        net.with_(max_iters=1).decode(s)


def test_convergence_error_carries_usable_state(rng: np.random.Generator):
    cbs = make_codebooks(FHRR, rng, num_factors=3, size=10, dim=1500)
    net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
    with pytest.raises(ConvergenceError) as excinfo:
        net.decode(FHRR.new(1500).data)
    assert excinfo.value.msg in str(excinfo.value)
    assert len(excinfo.value.estimates) == net.num_factors
    assert excinfo.value.iters >= 1


TRIALS = 30


@pytest.mark.parametrize("dim,size", [(500, 5), (1500, 10), (3000, 15)])
def test_success_rate_within_capacity(dim: int, size: int):

    def trial(seed: int, dim: int, size: int, num_factors: int) -> bool:
        np.random.seed(seed)
        rng = np.random.default_rng(seed=seed)
        cbs = make_codebooks(FHRR, rng, num_factors=num_factors, size=size, dim=dim)
        net = ResonatorNetwork(vsa=FHRR, codebooks=cbs, non_linearity=phasor_projection)
        indices = tuple(int(rng.integers(size)) for _ in range(num_factors))
        try:
            return net.decode(compose(FHRR, cbs, indices)).indices == indices
        except ConvergenceError:
            return False

    successes = sum(trial(seed, dim, size, num_factors=3) for seed in range(TRIALS))
    assert successes / TRIALS >= 0.9


def test_rhc_decodes_residue_number():
    RHC.set_moduli([3, 5, 7, 11])
    dim = 1500
    x = 52
    RHC.generate_basis(dim)
    codebooks = [
        np.stack([b**j for j in range(m)])
        for b, m in zip(RHC.basis, RHC.moduli, strict=True)
    ]
    result = ResonatorNetwork(
        vsa=RHC, codebooks=codebooks, non_linearity=phasor_projection
    ).decode(RHC.number(x, dim).data)
    assert result.indices == tuple(x % m for m in RHC.moduli)
