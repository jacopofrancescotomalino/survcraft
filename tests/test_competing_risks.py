import numpy as np
import pytest
import torch

from survcraft import adapters as ad
from survcraft import loss_modules as lm
from survcraft import survival_modules as sm


def competing_module(latents=None, grid=None):
    latents = latents or [sm.WeibullSurvivalModule(), sm.LogNormalSurvivalModule(), sm.ExponentialSurvivalModule()]
    grid = torch.linspace(0.0, 5.0, 40) if grid is None else grid
    return sm.CompetingRisksSurvivalModule(latents, grid)


def random_raw(module, n=6, dtype=torch.float32):
    return torch.randn(n, module.get_param_number(), dtype=dtype)


def test_cause_modes_have_trailing_cause_axis():
    module = competing_module()
    raw = random_raw(module)
    times = torch.tensor([0.5, 1.0, 2.0, 4.0])
    for mode in ["cif", "cs_density", "cs_hazard", "latent_survival"]:
        assert module(mode, raw, times).shape == (6, 4, 3)
        assert module(mode, raw, torch.tensor(1.5)).shape == (6, 3)
    for mode in ["failure", "survival", "density", "hazard"]:
        assert module(mode, raw, times).shape == (6, 4)


def test_single_risk_modules_are_one_cause_models():
    module = sm.WeibullSurvivalModule()
    raw = random_raw(module)
    times = torch.tensor([0.5, 1.0, 2.0])
    for cause_mode, mode in [("cif", "failure"), ("cs_density", "density"),
                             ("cs_hazard", "hazard"), ("latent_survival", "survival")]:
        out = module(cause_mode, raw, times)
        assert out.shape == (6, 3, 1)
        torch.testing.assert_close(out[..., 0], module(mode, raw, times))


def test_competing_identities():
    module = competing_module()
    raw = random_raw(module)
    times = torch.linspace(0.0, 6.0, 25)  # extends beyond the grid
    cif = module("cif", raw, times)
    survival = module("survival", raw, times)

    # all probability mass is either still at risk or assigned to one cause
    torch.testing.assert_close(cif.sum(-1) + survival, torch.ones_like(survival))
    assert (cif.diff(dim=1) >= -1e-7).all()
    torch.testing.assert_close(survival, module("latent_survival", raw, times).prod(-1))
    torch.testing.assert_close(module("density", raw, times), module("cs_density", raw, times).sum(-1))
    torch.testing.assert_close(module("hazard", raw, times), module("cs_hazard", raw, times).sum(-1))


def test_cif_handles_unsorted_and_duplicate_times():
    module = competing_module()
    raw = random_raw(module)
    times = torch.tensor([3.0, 0.0, 1.0, 3.0, 1.0])
    cif = module("cif", raw, times)
    order = torch.tensor([1, 2, 0])  # sorted unique: 0, 1, 3
    torch.testing.assert_close(cif[:, [1, 2, 0]], module("cif", raw, times[order]))
    torch.testing.assert_close(cif[:, 0], cif[:, 3])
    assert (cif[:, 1] == 0).all()


def test_cif_is_exact_for_constant_hazards():
    # with exponential latents F_k(t) = h_k / h (1 - exp(-h t)), even on a coarse grid
    module = competing_module([sm.ExponentialSurvivalModule() for _ in range(3)], torch.tensor([0.0, 1.0]))
    raw = random_raw(module, dtype=torch.float64)
    times = torch.tensor([0.3, 1.7, 4.0], dtype=torch.float64)
    rates = module("cs_hazard", raw, times)  # constant in time
    total = rates.sum(-1, keepdim=True)
    expected = rates / total * (1 - torch.exp(-total * times[:, None]))
    torch.testing.assert_close(module("cif", raw.double(), times), expected)


def test_cif_solves_the_ode():
    # dF_k/dt = h_k(t) (1 - sum_j F_j(t)) = cs_density, checked by finite differences
    module = competing_module(grid=torch.linspace(0.0, 4.0, 4000, dtype=torch.float64)).double()
    raw = random_raw(module, dtype=torch.float64)
    times = torch.linspace(0.5, 3.5, 7, dtype=torch.float64)
    step = 1e-3
    cif_up = module("cif", raw, times + step)
    cif_down = module("cif", raw, times - step)
    derivative = (cif_up - cif_down) / (2 * step)
    torch.testing.assert_close(derivative, module("cs_density", raw, times), rtol=1e-3, atol=1e-4)


def test_cif_gradients_are_finite():
    module = competing_module()
    raw = random_raw(module).requires_grad_()
    module("cif", raw, torch.tensor([0.0, 0.0, 1.0, 10.0])).sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_module_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="at least two"):
        sm.CompetingRisksSurvivalModule([sm.WeibullSurvivalModule()], torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="time_grid"):
        sm.CompetingRisksSurvivalModule([sm.WeibullSurvivalModule()] * 2, torch.tensor([0.5, 1.0]))


def torch_model(survival_module, n_features=3):
    return ad.TorchModel(torch.nn.Linear(n_features, survival_module.get_param_number()), survival_module)


@pytest.mark.parametrize("gpu_mode", [False, True])
def test_competing_loss_equals_full_likelihood_for_one_cause(gpu_mode):
    model = torch_model(sm.WeibullSurvivalModule())
    x = torch.randn(20, 3)
    times = torch.rand(20) * 3
    events = torch.rand(20) < 0.6
    expected = lm.FullLikelihoodLoss(censoring_alpha=0.7, gpu_mode=gpu_mode)(model, x, events, times)
    loss = lm.CompetingRisksLikelihoodLoss(censoring_alpha=0.7, gpu_mode=gpu_mode)
    torch.testing.assert_close(loss(model, x, events, times), expected)
    torch.testing.assert_close(loss(model, x, events.long(), times), expected)


@pytest.mark.parametrize("gpu_mode", [False, True])
def test_competing_loss_matches_manual_likelihood(gpu_mode):
    model = torch_model(competing_module())
    x = torch.randn(30, 3)
    times = torch.rand(30) * 3
    events = torch.randint(0, 4, (30,))

    idx = torch.arange(30)
    cs_density = model("cs_density", x, times)[idx, idx]  # (n, K)
    survival = model("survival", x, times)[idx, idx]
    observed = events > 0
    log_lik = torch.where(
        observed,
        torch.log(cs_density[idx, (events - 1).clamp_min(0)]),
        0.5 * torch.log(survival),
    )
    expected = -log_lik.mean()
    loss = lm.CompetingRisksLikelihoodLoss(censoring_alpha=0.5, gpu_mode=gpu_mode)
    torch.testing.assert_close(loss(model, x, events, times), expected)


def test_competing_loss_rejects_codes_beyond_causes():
    model = torch_model(competing_module())
    with pytest.raises(ValueError, match="causes"):
        lm.CompetingRisksLikelihoodLoss()(model, torch.randn(4, 3), torch.tensor([0, 1, 4, 2]), torch.rand(4))


def test_single_risk_losses_use_all_cause_indicator():
    model = torch_model(sm.WeibullSurvivalModule())
    x, times = torch.randn(20, 3), torch.rand(20) * 3
    codes = torch.randint(0, 3, (20,))
    for loss in [lm.FullLikelihoodLoss(), lm.PartialLikelihoodLoss(), lm.BrierLoss()]:
        torch.testing.assert_close(loss(model, x, codes, times), loss(model, x, codes > 0, times))


def test_adapter_builds_latents_from_their_own_cause():
    rng = np.random.default_rng(0)
    time = rng.exponential(size=200)
    event = rng.integers(0, 3, size=200)
    module = ad.CompetingRisksSurvivalAdapter(
        latents=[ad.StepExpSurvivalAdapter(breaks=4), ad.StepExpSurvivalAdapter(breaks=4)],
        time_grid=20,
    ).get_module(event, time)

    assert module.n_causes == 2
    for k, latent in enumerate(module.latents.values()):
        expected = ad.StepExpSurvivalAdapter(breaks=4).get_module(event == k + 1, time)
        torch.testing.assert_close(latent._get_time_breaks()[0], expected._get_time_breaks()[0])
    assert module.time_grid[0] == 0 and module.time_grid[-1] == pytest.approx(time.max())


def test_adapter_validation():
    time = np.ones(4)
    with pytest.raises(ValueError, match="latents are defined"):
        ad.CompetingRisksSurvivalAdapter().get_module(np.array([0, 1, 2, 3]), time)
    with pytest.raises(ValueError, match="at least two"):
        ad.CompetingRisksSurvivalAdapter(latents=[ad.WeibullSurvivalAdapter()]).get_module(None, None)
    with pytest.raises(ValueError, match="nonnegative integer"):
        ad.WeibullSurvivalAdapter().get_module(np.array([0, -1, 1, 0]), time)
    # without outcomes the grid falls back to linspace(0, 1, n)
    module = ad.CompetingRisksSurvivalAdapter(time_grid=5).get_module(None, None)
    torch.testing.assert_close(module.time_grid, torch.linspace(0, 1, 5))


def test_predictor_fits_competing_risks_targets():
    # latent exponential rates 1.0 and 0.5: cause 1 occurs first with probability 2/3
    rng = np.random.default_rng(0)
    n = 512
    X = rng.normal(size=(n, 2)).astype(np.float32)
    t1, t2 = rng.exponential(1.0, n), rng.exponential(2.0, n)
    time = np.minimum(np.minimum(t1, t2), 3.0)
    event = np.where(time == 3.0, 0, np.where(t1 < t2, 1, 2))
    y = np.array(list(zip(event, time)), dtype=[("event", "i8"), ("time", "f8")])

    predictor = ad.SurvivalPredictor(
        input=ad.FeedForwardNetAdapter(hidden_sizes=[8]),
        survival=ad.CompetingRisksSurvivalAdapter(
            latents=[ad.ExponentialSurvivalAdapter(), ad.ExponentialSurvivalAdapter()]
        ),
        loss=lm.CompetingRisksLikelihoodLoss(),
        epochs=60,
        batch_size=128,
        learning_rate=0.01,
    ).fit(X, y)

    times = np.array([0.5, 1.0, 20.0], dtype=np.float32)
    cif = predictor.predict("cif", X, times)
    survival = predictor.predict("survival", X, times)
    assert cif.shape == (n, 3, 2)
    np.testing.assert_allclose(cif.sum(-1) + survival, 1.0, atol=1e-5)
    # at a large time nearly all mass is assigned, split according to the hazard ratio
    assert cif[:, -1, 0].mean() == pytest.approx(2 / 3, abs=0.07)
