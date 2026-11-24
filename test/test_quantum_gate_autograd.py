"""Focused tests of the exact QuantumGateLayer source in processors.py.

Scope is deliberately isolated: processors.py currently imports SystemConfig,
UnifiedState, and ProcessingDimension that are absent from root config.py.
These tests therefore compile its unmodified QuantumGateLayer AST with real
torch/nn and the real utils.matrix_exponential callable. They neither invent
configuration stubs nor establish that processors.py or the application can
be imported. A passing result qualifies this class in isolation only.

Run explicitly from the repository root:

    python -m pytest -c test/pytest-gate.ini test/test_quantum_gate_autograd.py

The existing tests/ directory has its own unrelated import dependencies; this
focused command does not claim to execute or pass that full application suite.
"""

import ast
import importlib.util
import io
import math
from pathlib import Path

import pytest
from scipy.linalg import expm
import torch
from torch import nn
from torch.func import functional_call


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_exact_gate_class():
    """Compile the source class, retaining its real helper if still referenced."""
    processors_path = REPOSITORY_ROOT / "processors.py"
    tree = ast.parse(processors_path.read_text(encoding="utf-8"), filename=str(processors_path))
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QuantumGateLayer"
    ]
    assert len(classes) == 1, "expected exactly one top-level QuantumGateLayer"

    spec = importlib.util.spec_from_file_location(
        "_quantum_gate_real_repository_utils", REPOSITORY_ROOT / "utils.py"
    )
    assert spec is not None and spec.loader is not None
    real_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_utils)
    namespace = {
        "__name__": "_isolated_quantum_gate_source",
        "torch": torch,
        "nn": nn,
        "matrix_exponential": real_utils.matrix_exponential,
    }
    isolated = ast.Module(body=classes, type_ignores=[])
    exec(compile(isolated, filename=str(processors_path), mode="exec"), namespace)
    return namespace["QuantumGateLayer"]


QuantumGateLayer = _load_exact_gate_class()


def test_seeded_initialization_and_unitary_checkpoint_contract():
    with torch.random.fork_rng():
        torch.manual_seed(2718)
        expected = torch.randn(4, 4)
        torch.manual_seed(2718)
        gate = QuantumGateLayer(4)
    assert gate.dim == 4
    assert list(dict(gate.named_parameters())) == ["unitary"]
    assert list(gate.state_dict()) == ["unitary"]
    assert not list(gate.named_buffers())
    torch.testing.assert_close(gate.unitary, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_serialized_checkpoint_strict_load_and_scipy_output_parity(dtype):
    generator = torch.Generator().manual_seed(51)
    gate = QuantumGateLayer(3).to(dtype=dtype)
    with torch.no_grad():
        gate.unitary.copy_(0.15 * torch.randn(3, 3, generator=generator, dtype=dtype))
    checkpoint = io.BytesIO()
    torch.save(gate.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = QuantumGateLayer(3).to(dtype=dtype)
    result = restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    x = torch.randn(5, 3, generator=generator, dtype=dtype)
    reference_matrix = torch.from_numpy(expm(gate.unitary.detach().cpu().numpy()))
    expected = (reference_matrix @ x.T).T
    torch.testing.assert_close(restored(x), expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(restored(x), gate(x), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_actual_class_passes_parameter_and_input_gradcheck(dtype):
    generator = torch.Generator().manual_seed(62)
    gate = QuantumGateLayer(2).to(dtype=dtype)
    weight = (0.1 * torch.randn(2, 2, generator=generator, dtype=dtype)).requires_grad_()
    x = torch.randn(3, 2, generator=generator, dtype=dtype, requires_grad=True)

    def evaluate(parameter, inputs):
        # This invokes the source class's actual forward and parameter access;
        # the tested function is not a separately reconstructed exponential.
        return functional_call(gate, {"unitary": parameter}, (inputs,), strict=True)

    assert torch.autograd.gradcheck(
        evaluate, (weight, x), eps=1e-6, atol=1e-5, rtol=1e-3
    )


def test_trainable_upstream_and_gate_both_receive_gradients_and_update():
    generator = torch.Generator().manual_seed(73)
    upstream = nn.Linear(3, 3, bias=False, dtype=torch.float64)
    gate = QuantumGateLayer(3).double()
    with torch.no_grad():
        upstream.weight.copy_(torch.eye(3, dtype=torch.float64))
        gate.unitary.copy_(0.05 * torch.randn(3, 3, generator=generator, dtype=torch.float64))
    model = nn.Sequential(upstream, gate)
    x = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    before_upstream = upstream.weight.detach().clone()
    before_gate = gate.unitary.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    optimizer.zero_grad(set_to_none=True)
    (model(x) - target).square().mean().backward()

    assert upstream.weight.grad is not None
    assert torch.isfinite(upstream.weight.grad).all()
    assert upstream.weight.grad.abs().sum() > 0
    assert gate.unitary.grad is not None, "loss reached upstream but not the gate parameter"
    assert torch.isfinite(gate.unitary.grad).all()
    assert gate.unitary.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(upstream.weight, before_upstream)
    assert not torch.equal(gate.unitary, before_gate)


def test_raw_exponential_contract_can_amplify_norms():
    # exp(I) = e*I is a behavioral guard against silently introducing skew
    # subtraction or a norm-preserving architecture during a gradient repair.
    gate = QuantumGateLayer(3).double()
    with torch.no_grad():
        gate.unitary.copy_(torch.eye(3, dtype=torch.float64))
    x = torch.tensor([[1.0, -2.0, 3.0], [-0.5, 1.5, 2.0]], dtype=torch.float64)
    y = gate(x)
    torch.testing.assert_close(y, math.e * x, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(y.norm(dim=-1) / x.norm(dim=-1), torch.full((2,), math.e, dtype=x.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_cpu_dtype_device_and_finite_output_are_preserved(dtype):
    generator = torch.Generator().manual_seed(84)
    gate = QuantumGateLayer(3).to(device="cpu", dtype=dtype)
    with torch.no_grad():
        gate.unitary.copy_(0.1 * torch.randn(3, 3, generator=generator, dtype=dtype))
    x = torch.randn(4, 3, generator=generator, dtype=dtype)
    y = gate(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert torch.isfinite(y).all()
    reference_matrix = torch.from_numpy(expm(gate.unitary.detach().cpu().numpy()))
    torch.testing.assert_close(y, (reference_matrix @ x.T).T, atol=2e-5, rtol=2e-5)
