# AngerFlow

A Keras-style deep learning framework built **from scratch in NumPy**, for learning purposes.

**Goal:** `Sequential` + `Dense` + `compile / fit / predict` so that ANN problems (XOR, Iris, MNIST, regression) can be solved using AngerFlow alone.

> Not production-grade. The point is to understand every line: forward pass, backprop, optimizers, training loop.

---

## What are we building?

- A NumPy-only neural-network library with a Keras-like API
- Focus: **ANN / fully-connected networks**
- Solo project, learning first, roughly 4-6 weeks part-time

## Design decision

| Approach | Description | Status |
|---|---|---|
| **A. Layer-wise backward** | Every layer has its own `forward()` and `backward()` written manually | **Chosen** |
| **B. Autograd engine** | Computation graph with automatic gradients (micrograd/PyTorch style) | Stretch goal (Phase 9) |

**Conventions:** batch-first shapes `(batch, features)`, dtype `float32`.

---

## Target API

```python
import angerflow as af
from angerflow.layers import Dense, Dropout
from angerflow.optimizers import Adam

model = af.Sequential([
    Dense(64, activation="relu", input_dim=784),
    Dropout(0.2),
    Dense(10, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
model.evaluate(X_test, y_test)
model.predict(X_new)
```

---

## Folder structure

```
AngerFlow/
├── angerflow/
│   ├── __init__.py
│   ├── layers/          (base.py, dense.py, dropout.py, batchnorm.py)
│   ├── activations.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── initializers.py
│   ├── regularizers.py
│   ├── metrics.py
│   ├── callbacks.py
│   ├── models.py        (Sequential)
│   └── utils.py         (gradient_check, one_hot, train_test_split)
├── tests/
├── examples/            (xor.py, iris.py, mnist.py)
├── pyproject.toml
├── Project.md
└── README.md
```

---

## Roadmap (phase by phase)

Mark `[x]` as each item is done.

### Phase 0: Setup (1-2 days)
- [ ] Python venv, NumPy, pytest
- [ ] `pyproject.toml`, git repo
- [ ] Fix conventions (batch-first, float32)

### Phase 1: Core forward pass (3-4 days)
- [ ] `Layer` base class: `forward(x)`, `backward(grad)`, `params`, `grads`
- [ ] `Dense`: `y = xW + b`
- [ ] Activations: ReLU, Sigmoid, Tanh, Softmax, Linear (forward + derivative)
- [ ] `Sequential`: list of layers, forward chain
- **Milestone:** forward pass runs with random weights

### Phase 2: Loss + Backpropagation (5-6 days) ← most important
- [ ] Losses: MSE, MAE, BCE, CCE, Sparse CCE
- [ ] `backward()` for every layer: `dW = xᵀ·grad`, `db = sum(grad)`, `dx = grad·Wᵀ`
- [ ] Softmax + CCE combined gradient (`y_pred - y_true`)
- [ ] Gradient checking (numerical vs analytical)
- **Milestone:** gradient check passes (relative error < 1e-6)

### Phase 3: Optimizers (3 days)
- [ ] SGD
- [ ] SGD + Momentum
- [ ] RMSprop
- [ ] Adam (with bias correction)
- [ ] Base `Optimizer.update(params, grads)`
- **Milestone:** XOR problem solved

### Phase 4: Training loop (3-4 days)
- [ ] `compile()`, `fit()`: mini-batches, shuffle, epochs, validation split
- [ ] `History` object (loss/accuracy per epoch)
- [ ] `predict()`, `evaluate()`
- [ ] Progress output
- [ ] Metrics: accuracy, precision, recall, F1
- **Milestone:** Iris dataset 95%+ accuracy

### Phase 5: Initializers + Regularization (3-4 days)
- [ ] Initializers: Xavier/Glorot, He, zeros
- [ ] L1 / L2 regularizers (penalty in loss, gradient added in backward)
- [ ] Dropout (inverted dropout, train vs inference flag)
- [ ] BatchNorm (optional, tricky backward)
- **Milestone:** MNIST 97%+ accuracy

### Phase 6: Usability features (3-4 days)
- [ ] Callbacks: `EarlyStopping`, `ModelCheckpoint`, `LearningRateScheduler`
- [ ] `model.summary()` (layer-wise parameter counts)
- [ ] `save_weights` / `load_weights` (`.npz`)
- [ ] Learning rate decay

### Phase 7: Testing + Validation (2-3 days)
- [ ] pytest unit tests for every activation, loss, layer
- [ ] Cross-check against Keras/PyTorch (same weights + input, compare outputs and gradients)
- [ ] Benchmarks: XOR, Iris, Breast Cancer (binary), MNIST (multiclass), California Housing (regression)

### Phase 8: Packaging + Docs (2 days)
- [ ] README with examples and API docs
- [ ] `pip install -e .` works
- [ ] Push to GitHub
- [ ] Learning-curve plots in examples

### Phase 9: Stretch goals
- [ ] Autograd engine (`Tensor` with `.backward()`)
- [ ] `Conv2D`, `MaxPool`, `Flatten` (im2col)
- [ ] `RNN` / `LSTM`
- [ ] Functional API (multi-input/output)
- [ ] Speed: vectorization, CuPy for GPU

---

## Weekly plan

| Week | Phases | Result |
|---|---|---|
| 1 | 0, 1, 2 | Backprop + gradient check passes |
| 2 | 3, 4 | XOR + Iris train |
| 3 | 5 | MNIST 97%+ |
| 4 | 6, 7 | Callbacks, save/load, tests |
| 5 | 8 + stretch | Packaged library |

---

## Common pitfalls

- **Softmax overflow:** subtract `max(x)` before `exp`.
- **log(0):** `np.clip(y_pred, 1e-7, 1 - 1e-7)`.
- **Shape bugs:** bias gradient sums over `axis=0`; divide loss by batch size.
- **Dropout:** turn it off at inference.
- **Gradient check:** never skip it, it catches most bugs.
- **Zero weight init:** breaks symmetry, use Xavier/He.

---

## Progress log

| Date | What was done |
|---|---|
| 2026-11-20 | Project planned, roadmap written |
