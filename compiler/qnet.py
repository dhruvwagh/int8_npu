#!/usr/bin/env python3
"""
File: qnet.py
Description: MNIST CNN -> INT8 network for the NPU.

  1. Train a small float CNN (conv 1->8, pool, conv 8->16, pool, fc 784->10).
  2. Post-training quantization:
       - input pixels   x_q = round(x * 127)                  (scale 1/127, range 0..127)
       - weights        per-tensor symmetric int8, scale s_w = max|W| / 127
       - bias           int32 at scale s_in * s_w
       - activations    per-tensor int8, scale from a calibration percentile
       - requantization acc * RQ_MULT >> RQ_SHIFT (round half away from zero),
                        RQ_MULT / RQ_SHIFT approximate s_in * s_w / s_out
     Per-tensor weight scales because the NPU has one RQ_MULT per layer.
  3. Integer-only reference (int_forward) built on the same bit-exact
     epilogue as the hardware golden model. This IS the golden network.

conv layers are lowered to GEMM with im2col: K index = c*9 + ky*3 + kx
(PyTorch weight order), N index = y*W + x.
Author: Dhruv Wagh
"""

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from model.golden_npu import LayerCfg, wrap32  # noqa: E402
from model.golden_epilogue import epilogue_vector  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
CKPT = os.path.join(DATA_DIR, "mnist_cnn.pt")


# ----------------------------------------------------------------------------
# Float model
# ----------------------------------------------------------------------------
def build_float_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 7 * 7, 10),
    )


def load_mnist(train: bool):
    """Returns (images float32 [n,1,28,28] in [0,1], labels int64)."""
    import torchvision
    ds = torchvision.datasets.MNIST(DATA_DIR, train=train, download=True)
    x = ds.data.numpy().astype(np.float32)[:, None] / 255.0
    return x, ds.targets.numpy().astype(np.int64)


def train_float(epochs: int = 3, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    x, y = load_mnist(True)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    model = build_float_model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        model.train()
        for i in range(0, len(xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        print(f"  epoch {ep + 1}/{epochs}: last-batch loss {loss.item():.4f}")
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    return model


def load_float_model():
    import torch
    model = build_float_model()
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    return model


def float_accuracy(model, x, y) -> float:
    import torch
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).argmax(1).numpy()
    return float((pred == y).mean())


# ----------------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------------
@dataclass
class QLayer:
    name: str
    kind: str              # "conv" or "fc"
    Wq: np.ndarray         # (M, K) int8, GEMM A operand
    bq: np.ndarray         # (M,) int32
    cfg: LayerCfg          # requant config for the NPU (N filled per call)
    s_in: float
    s_w: float
    s_out: float
    in_shape: tuple        # (C, H, W) for conv, (K,) for fc


def requant_params(real_mult: float):
    """RQ_MULT / RQ_SHIFT with the largest shift <= 31 that keeps RQ_MULT < 2^31."""
    assert 0 < real_mult < 1, real_mult
    shift = 31
    while shift > 0 and round(real_mult * (1 << shift)) >= (1 << 31):
        shift -= 1
    return int(round(real_mult * (1 << shift))), shift


def quantize(model, x_calib: np.ndarray, pct: float = 99.99) -> List[QLayer]:
    import torch
    convs = [model[0], model[3]]
    fc = model[7]
    with torch.no_grad():
        t = torch.from_numpy(x_calib)
        a1 = model[1](model[0](t))              # conv1 + relu
        a2 = model[4](model[3](model[2](a1)))   # pool, conv2 + relu
        logits = model(t)
    acts = [a1.numpy(), a2.numpy(), logits.numpy()]

    s_in = 1.0 / 127.0
    layers = []
    specs = [("conv1", "conv", convs[0], (1, 28, 28), True),
             ("conv2", "conv", convs[1], (8, 14, 14), True),
             ("fc", "fc", fc, (784,), False)]
    for (name, kind, mod, in_shape, relu), act in zip(specs, acts):
        W = mod.weight.detach().numpy().reshape(mod.weight.shape[0], -1)
        b = mod.bias.detach().numpy()
        s_w = float(np.abs(W).max()) / 127.0
        Wq = np.clip(np.round(W / s_w), -127, 127).astype(np.int8)
        bq = np.round(b / (s_in * s_w)).astype(np.int64)
        assert np.all(np.abs(bq) < 2 ** 31)
        s_out = float(np.percentile(np.abs(act), pct)) / 127.0
        mult, shift = requant_params(s_in * s_w / s_out)
        cfg = LayerCfg(M=W.shape[0], N=0, K=W.shape[1], rq_mult=mult, rq_shift=shift,
                       relu_en=relu, bias_en=True, signed_act=True)
        layers.append(QLayer(name, kind, Wq, bq.astype(np.int32), cfg, s_in, s_w, s_out, in_shape))
        s_in = s_out   # maxpool keeps the scale
    return layers


# ----------------------------------------------------------------------------
# Integer reference (bit-exact with the NPU + host ops)
# ----------------------------------------------------------------------------
def quantize_input(x: np.ndarray) -> np.ndarray:
    """[0,1] float image (1,28,28) -> int8 in [0,127]."""
    return np.clip(np.round(x * 127.0), 0, 127).astype(np.int8)


def im2col3x3(x: np.ndarray) -> np.ndarray:
    """(C, H, W) int8, pad 1 -> (C*9, H*W)."""
    C, H, W = x.shape
    xp = np.zeros((C, H + 2, W + 2), dtype=np.int8)
    xp[:, 1:-1, 1:-1] = x
    cols = np.empty((C, 3, 3, H, W), dtype=np.int8)
    for ky in range(3):
        for kx in range(3):
            cols[:, ky, kx] = xp[:, ky:ky + H, kx:kx + W]
    return cols.reshape(C * 9, H * W)


def maxpool2(x: np.ndarray) -> np.ndarray:
    C, H, W = x.shape
    return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def gemm_layer(L: QLayer, B: np.ndarray) -> np.ndarray:
    """NPU layer: (M, K) x (K, N) + epilogue -> (M, N) int8."""
    acc = wrap32(L.Wq.astype(np.int64) @ B.astype(np.int64))
    c = L.cfg
    return epilogue_vector(acc, L.bq.astype(np.int64)[:, None], c.bias_en, c.rq_mult,
                           c.rq_shift, c.relu_en, c.signed_act)


def int_forward(layers: List[QLayer], x_q: np.ndarray):
    """x_q: (1, 28, 28) int8. Returns (c1 (8,784), c2 (16,196), logits (10,))."""
    c1 = gemm_layer(layers[0], im2col3x3(x_q))                  # (8, 784)
    p1 = maxpool2(c1.reshape(8, 28, 28))
    c2 = gemm_layer(layers[1], im2col3x3(p1))                   # (16, 196)
    p2 = maxpool2(c2.reshape(16, 14, 14))
    logits = gemm_layer(layers[2], p2.reshape(784, 1))[:, 0]    # (10,)
    return c1, c2, logits


def int_accuracy(layers, x, y, n=None):
    n = len(x) if n is None else n
    correct = 0
    for i in range(n):
        _, _, lg = int_forward(layers, quantize_input(x[i]))
        correct += int(np.argmax(lg) == y[i])
    return correct / n


def get_quantized_net(retrain: bool = False, verbose: bool = True):
    if retrain or not os.path.exists(CKPT):
        print("== Training float CNN")
        train_float()
    model = load_float_model()
    x_tr, _ = load_mnist(True)
    rng = np.random.default_rng(0)
    calib = x_tr[rng.choice(len(x_tr), 2000, replace=False)]
    layers = quantize(model, calib)
    if verbose:
        for L in layers:
            print(f"  {L.name:5s} M={L.cfg.M:2d} K={L.cfg.K:3d}  s_w={L.s_w:.5f} s_out={L.s_out:.5f}  "
                  f"RQ_MULT={L.cfg.rq_mult} RQ_SHIFT={L.cfg.rq_shift}")
    return model, layers


if __name__ == "__main__":
    model, layers = get_quantized_net(retrain="--retrain" in sys.argv)
    x_te, y_te = load_mnist(False)
    print(f"float accuracy (10k test): {100 * float_accuracy(model, x_te, y_te):.2f}%")
    print(f"int8  accuracy (10k test): {100 * int_accuracy(layers, x_te, y_te):.2f}%")
