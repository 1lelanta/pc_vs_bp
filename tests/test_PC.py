"""Unit tests for the Predictive Coding learner.

1. Hand-derived state and weight gradients == autograd on the energy F.
2. Inference behaves as the theory says (zero hidden energy at t=0,
   total energy decreases).
3. PC training never calls backward() and builds no autograd graph.
4. At equilibrium with a small target nudge, PC's weight updates equal a
   backprop pass seeded with PC's own (damped) output error.

Run from the project root:  python -m pytest -v
"""
import torch

from config import ModelCfg, PCCfg, TrainCfg
from data import to_one_hot
from models.predictive import InferenceResult, PCLearner

SIZES = (20, 16, 12, 10)     # small network, fast tests
B = 8                        # batch size


def make_learner(seed=0, double=True, steps=30, state_lr=0.05) -> PCLearner:
    torch.manual_seed(seed)
    m = PCLearner(ModelCfg(layer_sizes=SIZES), TrainCfg(),
                  PCCfg(inference_steps=steps, state_lr=state_lr))
    if double:
        m.layers.double()    # float64 makes the gradient comparison tight
    return m


def perturbed_states(m: PCLearner, seed=1):
    """Feedforward states with the hidden ones knocked off equilibrium, so
    every layer has a non-zero error and the gradients are non-trivial."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, SIZES[0], generator=g, dtype=torch.float64)
    target = torch.randn(B, SIZES[-1], generator=g, dtype=torch.float64)
    s = m._forward_states(x)
    hidden = [h + 0.3 * torch.randn(h.shape, generator=g, dtype=h.dtype)
              for h in s[1:-1]]
    return [s[0]] + hidden + [target]


def all_params(m: PCLearner):
    return [p for layer in m.layers for p in (layer.weight, layer.bias)]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0).item()


# ----------------------------------------------------------------------
# 1. Manual gradients == autograd on the energy
# ----------------------------------------------------------------------
def test_state_gradients_match_autograd():
    m = make_learner()
    states = perturbed_states(m)

    hidden = [s.clone().requires_grad_(True) for s in states[1:-1]]
    full = [states[0]] + hidden + [states[-1]]
    auto = torch.autograd.grad(m.energy(full), hidden)

    with torch.no_grad():
        errors, derivs = m._errors(states)
        manual = m.state_gradients(errors, derivs)

    # F is averaged over the batch; state_gradients is per-sample, so the
    # autograd result is smaller by exactly a factor of B.
    for a, g in zip(auto, manual):
        assert torch.allclose(a * B, g, atol=1e-10)


def test_weight_gradients_match_autograd():
    m = make_learner()
    states = perturbed_states(m)

    auto = torch.autograd.grad(m.energy(states), all_params(m))

    with torch.no_grad():
        errors, derivs = m._errors(states)
        result = InferenceResult(states, errors, derivs,
                                 torch.zeros(1, len(m.layers)))
        manual = m.weight_gradients(result)
    flat = [t for pair in manual for t in pair]

    assert len(auto) == len(flat)
    for a, g in zip(auto, flat):
        assert torch.allclose(a, g, atol=1e-10)


# ----------------------------------------------------------------------
# 2. Inference behaves as the theory says
# ----------------------------------------------------------------------
def test_inference_energy_behaviour():
    m = make_learner(steps=30)
    x = torch.randn(B, SIZES[0], dtype=torch.float64)
    y = torch.randint(0, 10, (B,))
    target = to_one_hot(y).double()

    trace = m.infer(x, target).energy_trace
    total = trace.sum(dim=1)

    assert trace.shape == (31, len(m.layers))
    # forward initialisation => hidden errors are exactly zero at t=0
    assert torch.allclose(trace[0, :-1], torch.zeros(len(m.layers) - 1,
                                                     dtype=torch.float64),
                          atol=1e-12)
    # gradient descent on F: total energy never increases
    assert (total[1:] <= total[:-1] + 1e-9).all()
    assert total[-1] < total[0]


# ----------------------------------------------------------------------
# 3. No backprop leaks into the PC path
# ----------------------------------------------------------------------
def test_pc_training_never_calls_backward(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("backward() was called during PC training")
    monkeypatch.setattr(torch.Tensor, "backward", boom)

    m = make_learner(double=False)
    x = torch.randn(B, SIZES[0])
    y = torch.randint(0, 10, (B,))
    before = [p.detach().clone() for p in m.layers.parameters()]

    m.train_step(x, y)

    for p in m.layers.parameters():
        assert p.grad is not None
        assert p.grad.grad_fn is None and not p.grad.requires_grad
    assert any(not torch.equal(b, p)
               for b, p in zip(before, m.layers.parameters()))


def test_inference_builds_no_autograd_graph():
    m = make_learner(double=False)
    x = torch.randn(B, SIZES[0])
    result = m.infer(x, to_one_hot(torch.randint(0, 10, (B,))))
    tensors = result.states + result.errors + result.derivs
    assert not any(t.requires_grad for t in tensors)


# ----------------------------------------------------------------------
# 4. Equilibrium PC == backprop recursion seeded with PC's own output error
# ----------------------------------------------------------------------
def test_pc_equilibrium_matches_backprop_seeded_with_its_own_error():
    """At equilibrium every hidden state satisfies dF/dx = 0, which is
    exactly the backprop recursion eps_l = W_{l+1}^T (f' * eps_{l+1}).
    So PC's weight updates equal a backprop pass whose output error is
    PC's equilibrium output error eps_L (NOT the raw error, which the
    hidden states have partly absorbed). With a tiny target nudge the
    states barely move, so the match is near-exact."""
    m = make_learner(steps=500, state_lr=0.1)
    x = torch.randn(B, SIZES[0], dtype=torch.float64)
    with torch.no_grad():
        out = m.forward(x)
    target = out + 0.001 * torch.randn_like(out)      # tiny nudge

    result = m.infer(x, target)

    # (a) inference really reached equilibrium
    with torch.no_grad():
        grads = m.state_gradients(result.errors, result.derivs)
    for g in grads:
        assert g.abs().max() < 1e-9

    # (b) PC weight updates == backprop seeded with eps_L
    pc_w = [gW for gW, _ in m.weight_gradients(result)]
    eps_L = result.errors[-1]
    weights = [layer.weight for layer in m.layers]
    bp_w = torch.autograd.grad(m.forward(x), weights,
                               grad_outputs=-eps_L / B)

    for layer_idx, (p, b) in enumerate(zip(pc_w, bp_w), start=1):
        assert cosine(p, b) > 0.999, f"layer {layer_idx} misaligned"