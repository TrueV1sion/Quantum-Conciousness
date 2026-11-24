# QuantumGateLayer autograd repair

## Problem and resulting behavior

`processors.QuantumGateLayer` passes its trainable `unitary` parameter to
`utils.matrix_exponential`, which detaches the tensor and computes the matrix
exponential through SciPy. A downstream loss can therefore update an upstream
layer while leaving `unitary.grad` unset.

The repair changes that call site to
`torch.linalg.matrix_exp(self.unitary)`. It preserves the constructor, seeded
initialization, `unitary` state-dict key, batch convention, and raw exponential
transformation. The operation remains generally nonorthogonal: a generator
equal to identity produces `e * I`. No skew subtraction, base buffer, or
checkpoint conversion is introduced.

The shared utility has other callers. This change is limited to the trainable
gate rather than changing the behavior of every utility consumer.

## Reproduce the focused regression

From the repository root, use Python 3.12 and an isolated environment. The
versions below reproduce the tested Linux CPU environment:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install 'numpy==2.3.5' 'scipy==1.17.0' 'pytest==9.2.0'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -c test/pytest-gate.ini test/test_quantum_gate_autograd.py
```

Use the explicit test file argument. The repository's default test discovery
and its application fixtures have separate import failures described below.

| Source under test | Focused result |
| --- | --- |
| Parent commit `45338bd6d871a56fa94d63649950180bf77d9fe0` | 3 failed, 6 passed |
| Parent plus the one-line repair | 9 passed |

The baseline failures are the real and complex parameter/input `gradcheck`
cases and the trainable-upstream/gate gradient-update case. The repaired class
passes all three. The other checks protect seeded initialization, strict
serialized checkpoint loading, agreement with SciPy, the raw exponential's
norm-scaling behavior, and float32/complex64 CPU dtype/device behavior.

These are newly generated checkpoint fixtures, not historical trained models.
PyTorch emits its standard complex-module warning during the complex cases.

### Test scope

The test compiles the exact `QuantumGateLayer` AST from `processors.py` and
executes it with real PyTorch objects and the real repository utility. It does
not reconstruct the layer formula, stub configuration types, or claim that
the full application imports successfully. This isolates the numerical
regression while retaining the actual source being changed.

## Existing application blockers

Both before and after this repair:

```text
python -c "import processors"
ImportError: cannot import name 'SystemConfig' from 'config'

python -m pytest --collect-only -q
ImportError: cannot import name 'PathwayConfig' from 'config'
```

These are pre-existing repository inconsistencies:

- Root `config.py` defines only `BridgeConfig`. The gate's containing module
  imports `SystemConfig`, `UnifiedState`, and `ProcessingDimension`, while
  `tests/conftest.py` also expects other missing configuration types.
- The similarly named types in `src/config.py` have different fields from
  those used by `EnhancedQuantumAgent`; redirecting imports alone is not a
  compatible repair.
- The agent calls bridge methods `establish_bridge` and
  `transfer_information`, whereas the bridge implements `transfer`. Its
  direction names and configuration contract also differ.
- The agent calls `PathwayOptimizer.optimize_pathway`, which does not exist.
  The implemented optimizer owns its policy network, not the agent's gate.
- The response tensor is constructed from the separate consciousness branch.
  The agent has no defined tensor objective that trains the gate, and its
  returned text and Python scalar confidence are not an autograd loss.

A subsequent application change must reconcile the intended contracts, pass a
CPU construction/forward smoke test, and demonstrate that the intended task
loss depends on the gate and that the actual optimizer updates its parameter.
The present change establishes layer-level differentiability and checkpoint
contract preservation; it does not establish end-to-end prediction or memory
quality improvements.
