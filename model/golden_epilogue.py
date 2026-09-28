#!/usr/bin/env python3
"""
File: golden_epilogue.py
Description: Bit-Exact Python Reference Model for NPU Epilogue
             Implements:
             1. 32-bit Accumulator + 32-bit Bias
             2. Signed 32x32 -> 64-bit Requantize Multiplier (RQ_MULT)
             3. Round Half Away From Zero Right Shift (RQ_SHIFT)
             4. ReLU Activation: max(0, x)
             5. Saturation Clamping: [-128, 127] for signed INT8 (or [0, 255] for unsigned)
Author: Dhruv Wagh
"""

import numpy as np


def epilogue_scalar(
    acc32: int,
    bias: int = 0,
    bias_en: bool = False,
    rq_mult: int = 1,
    rq_shift: int = 0,
    relu_en: bool = False,
    signed_act: bool = True
) -> int:
    """
    Bit-exact scalar computation for a single epilogue lane.
    All inputs and outputs are exact integers.
    """
    # 1. 32-bit accumulator + bias
    val = int(acc32)
    if bias_en:
        val += int(bias)

    # 2. 32x32 -> 64-bit signed multiplication
    prod = val * int(rq_mult)

    # 3. Round Half Away From Zero Right Shift
    if rq_shift > 0:
        half = 1 << (rq_shift - 1)
        if prod >= 0:
            shifted = (prod + half) >> rq_shift
        else:
            # Symmetric rounding away from zero for negative numbers
            # |prod| rounded away from zero: (-prod + half) >> rq_shift
            shifted = -((-prod + half) >> rq_shift)
    else:
        shifted = prod

    # 4. ReLU
    if relu_en:
        shifted = max(0, shifted)

    # 5. Saturation / Clamping
    if signed_act:
        out = min(127, max(-128, shifted))
    else:
        out = min(255, max(0, shifted))

    return out


def epilogue_vector(
    acc_vec: np.ndarray,
    bias_vec: np.ndarray = None,
    bias_en: bool = False,
    rq_mult: int = 1,
    rq_shift: int = 0,
    relu_en: bool = False,
    signed_act: bool = True
) -> np.ndarray:
    """
    Vectorized epilogue across an array of accumulators.
    """
    acc_flat = np.asarray(acc_vec, dtype=np.int64).flatten()
    if bias_vec is not None and bias_en:
        bias_flat = np.asarray(bias_vec, dtype=np.int64).flatten()
    else:
        bias_flat = np.zeros_like(acc_flat)

    out = np.zeros(len(acc_flat), dtype=np.int8 if signed_act else np.uint8)
    for i in range(len(acc_flat)):
        out[i] = epilogue_scalar(
            acc32=int(acc_flat[i]),
            bias=int(bias_flat[i]),
            bias_en=bias_en,
            rq_mult=rq_mult,
            rq_shift=rq_shift,
            relu_en=relu_en,
            signed_act=signed_act
        )
    return out.reshape(acc_vec.shape)


def generate_epilogue_test_suite(num_random: int = 500, seed: int = 42):
    """
    Generates a comprehensive suite of directed boundary and random test vectors.
    Returns a list of test case dictionaries.
    """
    rng = np.random.default_rng(seed)
    tests = []

    # -------------------------------------------------------------------------
    # 1. Boundary & Saturation Tests
    # -------------------------------------------------------------------------
    # Test values around clamping boundaries [-128, 127] with mult=1, shift=0
    for target in [-200, -129, -128, -127, -64, -2, -1, 0, 1, 2, 64, 126, 127, 128, 129, 200]:
        for bias in [0, -10, 10]:
            for bias_en in [False, True]:
                for relu_en in [False, True]:
                    for signed_act in [True]:
                        acc = target - (bias if bias_en else 0)
                        expected = epilogue_scalar(
                            acc32=acc, bias=bias, bias_en=bias_en,
                            rq_mult=1, rq_shift=0,
                            relu_en=relu_en, signed_act=signed_act
                        )
                        tests.append({
                            "name": f"Boundary_target_{target}_bias_{bias}_relu_{relu_en}",
                            "acc32": acc,
                            "bias": bias,
                            "bias_en": bias_en,
                            "rq_mult": 1,
                            "rq_shift": 0,
                            "relu_en": relu_en,
                            "signed_act": signed_act,
                            "expected": expected
                        })

    # -------------------------------------------------------------------------
    # 2. Rounding Half Away From Zero Symmetry Tests
    # -------------------------------------------------------------------------
    # Test exact halfway points (e.g. +2.5 -> +3, -2.5 -> -3)
    shifts = [1, 2, 4, 8, 16]
    for s in shifts:
        half_unit = 1 << (s - 1)
        # Integers K such that (K * 2^S + half_unit) / 2^S has fractional part exactly 0.5
        for k in [-5, -2, -1, 0, 1, 2, 5]:
            pos_prod = (k << s) + half_unit
            neg_prod = -pos_prod
            for relu in [False, True]:
                # Test positive halfway
                exp_pos = epilogue_scalar(pos_prod, 0, False, 1, s, relu, True)
                tests.append({
                    "name": f"Round_half_pos_k_{k}_s_{s}",
                    "acc32": pos_prod,
                    "bias": 0,
                    "bias_en": False,
                    "rq_mult": 1,
                    "rq_shift": s,
                    "relu_en": relu,
                    "signed_act": True,
                    "expected": exp_pos
                })
                # Test negative halfway
                exp_neg = epilogue_scalar(neg_prod, 0, False, 1, s, relu, True)
                tests.append({
                    "name": f"Round_half_neg_k_{k}_s_{s}",
                    "acc32": neg_prod,
                    "bias": 0,
                    "bias_en": False,
                    "rq_mult": 1,
                    "rq_shift": s,
                    "relu_en": relu,
                    "signed_act": True,
                    "expected": exp_neg
                })

    # -------------------------------------------------------------------------
    # 3. Dynamic Range & Random Cases
    # -------------------------------------------------------------------------
    for i in range(num_random):
        acc = int(rng.integers(-2**30, 2**30))
        bias = int(rng.integers(-2**20, 2**20))
        bias_en = bool(rng.choice([True, False]))
        rq_mult = int(rng.integers(-2**20, 2**20))
        rq_shift = int(rng.integers(0, 24))
        relu_en = bool(rng.choice([True, False]))
        signed_act = True

        expected = epilogue_scalar(
            acc32=acc, bias=bias, bias_en=bias_en,
            rq_mult=rq_mult, rq_shift=rq_shift,
            relu_en=relu_en, signed_act=signed_act
        )
        tests.append({
            "name": f"Random_{i}",
            "acc32": acc,
            "bias": bias,
            "bias_en": bias_en,
            "rq_mult": rq_mult,
            "rq_shift": rq_shift,
            "relu_en": relu_en,
            "signed_act": signed_act,
            "expected": expected
        })

    return tests


def export_epilogue_vectors(filename: str, num_random: int = 1000, seed: int = 123):
    """
    Exports test suite to a structured text file for consumption by SystemVerilog testbenches.
    Format per line:
    acc32(hex32) bias(hex32) bias_en rq_mult(hex32) rq_shift(hex2) relu_en signed_act expected(hex2)
    """
    tests = generate_epilogue_test_suite(num_random=num_random, seed=seed)
    with open(filename, "w") as f:
        f.write(f"# Epilogue Test Vectors (Total: {len(tests)})\n")
        f.write("# Format: acc32 bias bias_en rq_mult rq_shift relu_en signed_act expected\n")
        for t in tests:
            acc_u32 = t["acc32"] & 0xFFFF_FFFF
            bias_u32 = t["bias"] & 0xFFFF_FFFF
            b_en = 1 if t["bias_en"] else 0
            mult_u32 = t["rq_mult"] & 0xFFFF_FFFF
            shift_u8 = t["rq_shift"] & 0xFF
            r_en = 1 if t["relu_en"] else 0
            s_act = 1 if t["signed_act"] else 0
            exp_u8 = t["expected"] & 0xFF

            f.write(f"{acc_u32:08x} {bias_u32:08x} {b_en:d} {mult_u32:08x} {shift_u8:02x} {r_en:d} {s_act:d} {exp_u8:02x}\n")

    print(f"Exported {len(tests)} epilogue test vectors to {filename}")
    return len(tests)


if __name__ == "__main__":
    export_epilogue_vectors("tb/vectors/epilogue_vectors.txt", num_random=1000)
