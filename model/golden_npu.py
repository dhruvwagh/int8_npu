#!/usr/bin/env python3
"""
File: golden_npu.py
Description: Host-side data packing and bit-exact reference for one NPU layer.

Tensor memory layout (the contract between compiler and hardware, SPEC section 9):
  A (weights, M x K):     panel t holds rows 16t..16t+15. Word (t, k) = A[16t+r][k]
                          in byte r. Address a_base + (t*K + k)*16.
  B (activations, K x N): panel u holds cols 16u..16u+15. Word (u, k) = B[k][16u+c]
                          in byte c. Address b_base + (u*K + k)*16.
  Bias (int32, M):        little-endian, 16 per panel (64 B). Address bias_base + m*4.
  C (outputs, M x N):     word (t, n) = C[16t+r][n] in byte r. Address c_base + (t*N + n)*16.
Panels are zero-padded to 16 rows/cols; padded output rows are written too.

Byte r of a 128-bit word sits at bits [8r+7:8r], so a word's hex string lists
byte 15 first.
Author: Dhruv Wagh
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from model.golden_epilogue import epilogue_vector

ARR = 16          # ARR_M == ARR_N
WORD_B = 16       # bytes per 128-bit memory word


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def pad_to(x: int, m: int = ARR) -> int:
    return ceil_div(x, m) * m


# ----------------------------------------------------------------------------
# Packing
# ----------------------------------------------------------------------------
def pack_a(A: np.ndarray) -> bytes:
    """Weights (M, K) int8 -> panel layout."""
    M, K = A.shape
    Ap = np.zeros((pad_to(M), K), dtype=np.int8)
    Ap[:M] = A
    # (t, r, k) -> (t, k, r)
    return Ap.reshape(-1, ARR, K).transpose(0, 2, 1).astype(np.int8).tobytes()


def pack_b(B: np.ndarray) -> bytes:
    """Activations (K, N) int8 -> panel layout."""
    K, N = B.shape
    Bp = np.zeros((K, pad_to(N)), dtype=np.int8)
    Bp[:, :N] = B
    # (k, u, c) -> (u, k, c)
    return Bp.reshape(K, -1, ARR).transpose(1, 0, 2).astype(np.int8).tobytes()


def pack_bias(bias: np.ndarray, M: int) -> bytes:
    bp = np.zeros(pad_to(M), dtype="<i4")
    bp[:M] = bias
    return bp.tobytes()


def pack_c(Cpad: np.ndarray) -> bytes:
    """Padded outputs (M_pad, N) int8 -> output layout."""
    Mp, N = Cpad.shape
    return Cpad.reshape(-1, ARR, N).transpose(0, 2, 1).astype(np.int8).tobytes()


def unpack_c(raw: bytes, M: int, N: int) -> np.ndarray:
    Mp = pad_to(M)
    arr = np.frombuffer(raw, dtype=np.int8)[: Mp * N].reshape(-1, N, ARR)
    return arr.transpose(0, 2, 1).reshape(Mp, N)[:M]


# ----------------------------------------------------------------------------
# Reference
# ----------------------------------------------------------------------------
@dataclass
class LayerCfg:
    M: int
    N: int
    K: int
    rq_mult: int = 1
    rq_shift: int = 0
    relu_en: bool = False
    bias_en: bool = False
    signed_act: bool = True

    @property
    def flags(self) -> int:
        return int(self.relu_en) | (int(self.bias_en) << 1) | (int(self.signed_act) << 2)


def wrap32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64) & 0xFFFFFFFF
    return np.where(x >= 1 << 31, x - (1 << 32), x)


def layer_golden(A: np.ndarray, B: np.ndarray, bias: np.ndarray, cfg: LayerCfg) -> np.ndarray:
    """Returns the padded (M_pad, N) INT8 output exactly as the hardware writes it."""
    Mp = pad_to(cfg.M)
    Ap = np.zeros((Mp, cfg.K), dtype=np.int64)
    Ap[: cfg.M] = A
    bp = np.zeros(Mp, dtype=np.int64)
    bp[: cfg.M] = bias
    acc = wrap32(Ap @ B.astype(np.int64))
    out = epilogue_vector(acc, bp[:, None], cfg.bias_en, cfg.rq_mult, cfg.rq_shift,
                          cfg.relu_en, cfg.signed_act)
    # Hardware stores the low byte; view unsigned-mode results as raw int8 bytes
    return out.view(np.int8)


def choose_requant(acc_absmax: int, target: int = 100, shift: int = 24):
    """Fixed-point multiplier so that acc_absmax maps to about `target`."""
    acc_absmax = max(int(acc_absmax), 1)
    mult = int(round(target / acc_absmax * (1 << shift)))
    mult = max(1, min(mult, (1 << 31) - 1))
    return mult, shift


# ----------------------------------------------------------------------------
# Memory image
# ----------------------------------------------------------------------------
class MemImage:
    """Byte-addressed tensor memory made of 128-bit words."""

    def __init__(self, size_bytes: int):
        assert size_bytes % WORD_B == 0
        self.data = bytearray(size_bytes)
        self.brk = 0

    def alloc(self, nbytes: int, align: int = 64) -> int:
        base = pad_to(self.brk, align)
        self.brk = base + pad_to(nbytes, WORD_B)
        assert self.brk <= len(self.data), "memory image too small"
        return base

    def place(self, payload: bytes, align: int = 64) -> int:
        base = self.alloc(len(payload), align)
        self.data[base: base + len(payload)] = payload
        return base

    def fill(self, base: int, nbytes: int, value: int) -> None:
        self.data[base: base + nbytes] = bytes([value]) * nbytes

    def write(self, base: int, payload: bytes) -> None:
        self.data[base: base + len(payload)] = payload

    def used_words(self) -> int:
        return self.brk // WORD_B

    def to_hex_lines(self, nwords: Optional[int] = None) -> List[str]:
        n = self.used_words() if nwords is None else nwords
        return [self.data[w * WORD_B:(w + 1) * WORD_B][::-1].hex() for w in range(n)]

    def copy(self) -> "MemImage":
        m = MemImage(len(self.data))
        m.data[:] = self.data
        m.brk = self.brk
        return m


@dataclass
class LayerJob:
    """One layer placed in memory, with its expected post-run image."""
    cfg: LayerCfg
    a_base: int
    b_base: int
    c_base: int
    bias_base: int
    mem_in: MemImage
    mem_exp: MemImage
    C: np.ndarray = field(repr=False, default=None)


POISON = 0xA5  # C region is pre-filled with this so missed writes are caught


def build_layer_job(A, B, bias, cfg: LayerCfg, mem_bytes: int = 1 << 20) -> LayerJob:
    mem = MemImage(mem_bytes)
    a_base = mem.place(pack_a(A))
    b_base = mem.place(pack_b(B))
    bias_base = mem.place(pack_bias(bias, cfg.M))
    c_bytes = pad_to(cfg.M) * cfg.N
    c_base = mem.alloc(c_bytes)
    mem.fill(c_base, c_bytes, POISON)
    # Guard word after C catches overruns
    guard = mem.alloc(WORD_B)
    mem.fill(guard, WORD_B, 0x5A)

    Cpad = layer_golden(A, B, bias, cfg)
    exp = mem.copy()
    exp.write(c_base, pack_c(Cpad))
    return LayerJob(cfg, a_base, b_base, c_base, bias_base, mem, exp, Cpad[: cfg.M])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M, N, K = 20, 37, 150
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    B = rng.integers(-128, 128, (K, N), dtype=np.int8)
    # Round-trip checks
    Cpad = rng.integers(-128, 128, (pad_to(M), N), dtype=np.int8)
    assert np.array_equal(unpack_c(pack_c(Cpad), M, N), Cpad[:M])
    raw = pack_a(A)
    assert raw[(1 * K + 3) * 16 + 2] == np.uint8(A[16 + 2, 3])
    raw = pack_b(B)
    assert raw[(2 * K + 5) * 16 + 4] == np.uint8(B[5, 32 + 4])
    print("golden_npu packing self-test passed")
