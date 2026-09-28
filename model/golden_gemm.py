"""
Golden Python/NumPy Reference Model for INT8 Systolic Array GEMM.

Provides:
1. Pure functional golden reference: numpy.matmul with signed INT8 -> INT32.
2. Cycle-accurate systolic array simulator matching npu_array + npu_skew:
   - Output-stationary dataflow
   - Triangular skew registers on activations (west) and weights (north)
   - PE operand forwarding latency (1 cycle per hop)
   - Cycle-by-cycle accumulator inspection
   - Fill and drain latency verification
3. Test vector generation for 4x4, 8x8, and 16x16 arrays.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


def int8_gemm_golden(
    A: np.ndarray,
    B: np.ndarray,
    acc_w: int = 32
) -> np.ndarray:
    """
    Bit-exact functional reference for INT8 matrix multiplication.
    A: (M, K) signed int8
    B: (K, N) signed int8
    Returns: C (M, N) signed int of bit width acc_w
    """
    assert A.dtype == np.int8 or np.issubdtype(A.dtype, np.signedinteger)
    assert B.dtype == np.int8 or np.issubdtype(B.dtype, np.signedinteger)
    assert A.shape[1] == B.shape[0], f"Shape mismatch: A {A.shape}, B {B.shape}"

    # Standard matrix multiplication in signed 32-bit integer arithmetic
    C = np.matmul(A.astype(np.int64), B.astype(np.int64))

    # Two's complement masking to acc_w
    mask = (1 << acc_w) - 1
    sign_bit = 1 << (acc_w - 1)
    C_wrapped = (C & mask)
    C_signed = np.where(C_wrapped >= sign_bit, C_wrapped - (1 << acc_w), C_wrapped)
    return C_signed.astype(np.int32 if acc_w <= 32 else np.int64)


class CycleAccurateSystolicArray:
    """
    Cycle-accurate model of npu_skew + npu_array.
    Mirrors the exact register states of the SystemVerilog hardware.
    """

    def __init__(self, arr_m: int = 4, arr_n: int = 4, acc_w: int = 32):
        self.arr_m = arr_m
        self.arr_n = arr_n
        self.acc_w = acc_w
        self.reset()

    def reset(self) -> None:
        """Reset all hardware registers."""
        # Skew registers:
        # row i has i delay stages
        self.a_skew = [[0] * i for i in range(self.arr_m)]
        # col j has j delay stages
        self.b_skew = [[0] * j for j in range(self.arr_n)]
        # clr skew: row i delayed by i, then col j delayed by j
        self.clr_skew = [[0] * (i + j) for i in range(self.arr_m) for j in range(self.arr_n)]

        # PE internal state:
        # a_reg[i][j]: forwarded east
        self.pe_a = np.zeros((self.arr_m, self.arr_n), dtype=np.int8)
        # b_reg[i][j]: forwarded south
        self.pe_b = np.zeros((self.arr_m, self.arr_n), dtype=np.int8)
        # accumulators
        self.pe_acc = np.zeros((self.arr_m, self.arr_n), dtype=np.int64)

    def step(
        self,
        en: bool,
        clr_in: bool,
        a_in: np.ndarray,  # shape (M,)
        b_in: np.ndarray   # shape (N,)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance one clock cycle.
        Returns: (pe_a, pe_b, pe_acc)
        """
        if not en:
            return self.pe_a.copy(), self.pe_b.copy(), self.pe_acc.copy()

        M, N = self.arr_m, self.arr_n

        # 1. Advance skew registers
        a_array_in = np.zeros(M, dtype=np.int8)
        for i in range(M):
            if i == 0:
                a_array_in[0] = a_in[0]
            else:
                a_array_in[i] = self.a_skew[i][-1] if len(self.a_skew[i]) > 0 else a_in[i]
                # Shift
                for s in range(len(self.a_skew[i]) - 1, 0, -1):
                    self.a_skew[i][s] = self.a_skew[i][s - 1]
                self.a_skew[i][0] = int(a_in[i])

        b_array_in = np.zeros(N, dtype=np.int8)
        for j in range(N):
            if j == 0:
                b_array_in[0] = b_in[0]
            else:
                b_array_in[j] = self.b_skew[j][-1] if len(self.b_skew[j]) > 0 else b_in[j]
                # Shift
                for s in range(len(self.b_skew[j]) - 1, 0, -1):
                    self.b_skew[j][s] = self.b_skew[j][s - 1]
                self.b_skew[j][0] = int(b_in[j])

        # Clr skew for each (i, j): delay of (i + j) cycles
        clr_pe = np.zeros((M, N), dtype=bool)
        idx = 0
        for i in range(M):
            for j in range(N):
                delay = i + j
                if delay == 0:
                    clr_pe[i, j] = clr_in
                else:
                    clr_pe[i, j] = bool(self.clr_skew[idx][-1])
                    for s in range(delay - 1, 0, -1):
                        self.clr_skew[idx][s] = self.clr_skew[idx][s - 1]
                    self.clr_skew[idx][0] = 1 if clr_in else 0
                idx += 1

        # 2. Compute PE inputs for this cycle
        # PE(i, j) receives a from PE(i, j-1) or a_array_in[i]
        # PE(i, j) receives b from PE(i-1, j) or b_array_in[j]
        pe_a_in = np.zeros((M, N), dtype=np.int8)
        pe_b_in = np.zeros((M, N), dtype=np.int8)

        for i in range(M):
            for j in range(N):
                pe_a_in[i, j] = a_array_in[i] if j == 0 else self.pe_a[i, j - 1]
                pe_b_in[i, j] = b_array_in[j] if i == 0 else self.pe_b[i - 1, j]

        # 3. PE state update
        next_acc = self.pe_acc.copy()
        for i in range(M):
            for j in range(N):
                prod = int(pe_a_in[i, j]) * int(pe_b_in[i, j])
                if clr_pe[i, j]:
                    next_acc[i, j] = prod
                else:
                    next_acc[i, j] = self.pe_acc[i, j] + prod

                # Wrap accumulator
                mask = (1 << self.acc_w) - 1
                sign_bit = 1 << (self.acc_w - 1)
                val = next_acc[i, j] & mask
                if val >= sign_bit:
                    val -= (1 << self.acc_w)
                next_acc[i, j] = val

        self.pe_a = pe_a_in.copy()
        self.pe_b = pe_b_in.copy()
        self.pe_acc = next_acc.copy()

        return self.pe_a.copy(), self.pe_b.copy(), self.pe_acc.copy()

    def run_gemm(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Run a complete GEMM of shape (M, K) x (K, N) through the cycle-accurate array.
        Returns: (final_C, stats)
        """
        M, K = A.shape
        K_b, N = B.shape
        assert M == self.arr_m and N == self.arr_n and K == K_b

        self.reset()

        # Hand calculation latencies:
        # Fill latency to PE(M-1, N-1): (M-1) + (N-1) = M + N - 2
        fill_latency = (M - 1) + (N - 1)
        # Total compute cycles until PE(M-1, N-1) finishes reduction:
        # K cycles of feeding operands + (M + N - 2) cycles of propagation
        total_compute_cycles = K + fill_latency
        drain_latency = N

        stats = {
            "fill_latency": fill_latency,
            "total_compute_cycles": total_compute_cycles,
            "drain_latency": drain_latency,
            "macs": M * N * K
        }

        # Feed inputs for K cycles
        for k in range(K):
            clr_in = (k == 0)
            a_col = A[:, k]
            b_row = B[k, :]
            self.step(en=True, clr_in=clr_in, a_in=a_col, b_in=b_row)

        # Flush remaining pipeline hops (M + N - 2 cycles) with zero inputs
        zero_a = np.zeros(M, dtype=np.int8)
        zero_b = np.zeros(N, dtype=np.int8)
        for _ in range(fill_latency):
            self.step(en=True, clr_in=False, a_in=zero_a, b_in=zero_b)

        return self.pe_acc.copy(), stats


def generate_gemm_test_case(
    arr_m: int = 4,
    arr_n: int = 4,
    k: int = 4,
    seed: int = 42
) -> Dict[str, Any]:
    """Generate a random GEMM test case and run both golden and cycle-accurate models."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, size=(arr_m, k), dtype=np.int8)
    B = rng.integers(-128, 128, size=(k, arr_n), dtype=np.int8)

    C_golden = int8_gemm_golden(A, B)

    sim = CycleAccurateSystolicArray(arr_m=arr_m, arr_n=arr_n)
    C_systolic, stats = sim.run_gemm(A, B)

    assert np.array_equal(C_golden, C_systolic), (
        f"Cycle-accurate systolic model mismatch!\n"
        f"Golden:\n{C_golden}\nSystolic:\n{C_systolic}"
    )

    return {
        "A": A,
        "B": B,
        "C_golden": C_golden,
        "stats": stats
    }


def export_gemm_suite(filepath: str, num_tests: int = 20, arr_m: int = 4, arr_n: int = 4, seed: int = 42) -> None:
    """
    Export multiple GEMM test cases to a text file for Verilog testbench ingestion.
    Format per test:
    TEST <test_id> <M> <N> <K>
    A: <hex bytes space-separated>
    B: <hex bytes space-separated>
    C: <hex 32-bit words space-separated>
    """
    rng = np.random.default_rng(seed)
    with open(filepath, "w") as f:
        f.write(f"# GEMM Test Suite: ARR_M={arr_m}, ARR_N={arr_n}, num_tests={num_tests}\n")

        # Test 0: Identity / simple
        A0 = np.eye(arr_m, dtype=np.int8) * 10
        B0 = np.eye(arr_n, dtype=np.int8) * 5
        C0 = int8_gemm_golden(A0, B0)
        _write_test_case(f, 0, A0, B0, C0)

        # Test 1: Max positive and negative extremes
        A1 = np.full((arr_m, 4), 127, dtype=np.int8)
        B1 = np.full((4, arr_n), -128, dtype=np.int8)
        C1 = int8_gemm_golden(A1, B1)
        _write_test_case(f, 1, A1, B1, C1)

        # Remaining tests: Random M x K x N with varying K
        for t in range(2, num_tests):
            k = int(rng.choice([1, 2, 4, 8, 16]))
            A = rng.integers(-128, 128, size=(arr_m, k), dtype=np.int8)
            B = rng.integers(-128, 128, size=(k, arr_n), dtype=np.int8)
            C = int8_gemm_golden(A, B)
            _write_test_case(f, t, A, B, C)


def _write_test_case(f, test_id: int, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    M, K = A.shape
    K_b, N = B.shape
    f.write(f"TEST {test_id} {M} {N} {K}\n")
    a_hex = " ".join(f"{int(val) & 0xFF:02x}" for val in A.flatten())
    f.write(f"A {a_hex}\n")
    b_hex = " ".join(f"{int(val) & 0xFF:02x}" for val in B.flatten())
    f.write(f"B {b_hex}\n")
    c_hex = " ".join(f"{int(val) & 0xFFFFFFFF:08x}" for val in C.flatten())
    f.write(f"C {c_hex}\n")


def export_parameterized_gemm_suite(
    filepath: str,
    num_tests: int = 30,
    arr_m: int = 16,
    arr_n: int = 16,
    allow_subtiles: bool = True,
    seed: int = 42
) -> None:
    """
    Export test cases for full-tile and partial/sub-tile GEMMs (M <= ARR_M, N <= ARR_N, K <= 64).
    Unused rows/cols in the array are zero-padded.
    Format per test:
    TEST <test_id> <M_active> <N_active> <K> <ARR_M> <ARR_N>
    A: <ARR_M * K hex bytes>
    B: <K * ARR_N hex bytes>
    C: <ARR_M * ARR_N hex 32-bit words>
    """
    rng = np.random.default_rng(seed)
    with open(filepath, "w") as f:
        f.write(f"# Parameterized GEMM Suite: ARR_M={arr_m}, ARR_N={arr_n}, num_tests={num_tests}\n")

        # Test 0: Full tile Identity
        A0 = np.eye(arr_m, dtype=np.int8) * 7
        B0 = np.eye(arr_n, dtype=np.int8) * 3
        C0 = int8_gemm_golden(A0, B0)
        _write_parameterized_test_case(f, 0, arr_m, arr_n, arr_m, arr_m, arr_n, A0, B0, C0)

        # Test 1: Full tile extreme corner values (-128 * 127)
        A1 = np.full((arr_m, 8), 127, dtype=np.int8)
        B1 = np.full((8, arr_n), -128, dtype=np.int8)
        C1 = int8_gemm_golden(A1, B1)
        _write_parameterized_test_case(f, 1, arr_m, arr_n, 8, arr_m, arr_n, A1, B1, C1)

        # Test 2: Degenerate case N=1 (Vector product, FC layer like MNIST FC: 10 x 1 x K)
        M_act, N_act, K_val = 10, 1, 16
        A2 = np.zeros((arr_m, K_val), dtype=np.int8)
        B2 = np.zeros((K_val, arr_n), dtype=np.int8)
        A2[:M_act, :] = rng.integers(-128, 128, size=(M_act, K_val), dtype=np.int8)
        B2[:, :N_act] = rng.integers(-128, 128, size=(K_val, N_act), dtype=np.int8)
        C2 = int8_gemm_golden(A2, B2)
        _write_parameterized_test_case(f, 2, M_act, N_act, K_val, arr_m, arr_n, A2, B2, C2)

        # Test 3: Degenerate case M=1 (Single channel output)
        M_act, N_act, K_val = 1, 16, 12
        A3 = np.zeros((arr_m, K_val), dtype=np.int8)
        B3 = np.zeros((K_val, arr_n), dtype=np.int8)
        A3[:M_act, :] = rng.integers(-128, 128, size=(M_act, K_val), dtype=np.int8)
        B3[:, :N_act] = rng.integers(-128, 128, size=(K_val, N_act), dtype=np.int8)
        C3 = int8_gemm_golden(A3, B3)
        _write_parameterized_test_case(f, 3, M_act, N_act, K_val, arr_m, arr_n, A3, B3, C3)

        # Test 4: Degenerate case K=1
        M_act, N_act, K_val = arr_m, arr_n, 1
        A4 = rng.integers(-128, 128, size=(arr_m, 1), dtype=np.int8)
        B4 = rng.integers(-128, 128, size=(1, arr_n), dtype=np.int8)
        C4 = int8_gemm_golden(A4, B4)
        _write_parameterized_test_case(f, 4, M_act, N_act, K_val, arr_m, arr_n, A4, B4, C4)

        # Test 5: Conv1-like shape (M=8 < ARR_M, K=9)
        M_act, N_act, K_val = 8, 16, 9
        A5 = np.zeros((arr_m, K_val), dtype=np.int8)
        B5 = np.zeros((K_val, arr_n), dtype=np.int8)
        A5[:M_act, :] = rng.integers(-128, 128, size=(M_act, K_val), dtype=np.int8)
        B5[:, :N_act] = rng.integers(-128, 128, size=(K_val, N_act), dtype=np.int8)
        C5 = int8_gemm_golden(A5, B5)
        _write_parameterized_test_case(f, 5, M_act, N_act, K_val, arr_m, arr_n, A5, B5, C5)

        # Remaining tests: Random sub-tiles and full tiles with diverse K
        for t in range(6, num_tests):
            if allow_subtiles:
                M_act = int(rng.integers(1, arr_m + 1))
                N_act = int(rng.integers(1, arr_n + 1))
            else:
                M_act = arr_m
                N_act = arr_n
            K_val = int(rng.choice([1, 2, 3, 5, 8, 13, 16, 24, 32, 48, 64]))

            A = np.zeros((arr_m, K_val), dtype=np.int8)
            B = np.zeros((K_val, arr_n), dtype=np.int8)
            A[:M_act, :] = rng.integers(-128, 128, size=(M_act, K_val), dtype=np.int8)
            B[:, :N_act] = rng.integers(-128, 128, size=(K_val, N_act), dtype=np.int8)
            C = int8_gemm_golden(A, B)
            _write_parameterized_test_case(f, t, M_act, N_act, K_val, arr_m, arr_n, A, B, C)


def _write_parameterized_test_case(
    f, test_id: int, M_act: int, N_act: int, K: int, ARR_M: int, ARR_N: int,
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> None:
    f.write(f"TEST {test_id} {M_act} {N_act} {K} {ARR_M} {ARR_N}\n")
    a_hex = " ".join(f"{int(val) & 0xFF:02x}" for val in A.flatten())
    f.write(f"A {a_hex}\n")
    b_hex = " ".join(f"{int(val) & 0xFF:02x}" for val in B.flatten())
    f.write(f"B {b_hex}\n")
    c_hex = " ".join(f"{int(val) & 0xFFFFFFFF:08x}" for val in C.flatten())
    f.write(f"C {c_hex}\n")


if __name__ == "__main__":
    import os
    import sys

    out_dir = os.path.join(os.path.dirname(__file__), "..", "tb", "vectors")
    os.makedirs(out_dir, exist_ok=True)

    # Export 4x4 suite
    out_file_4 = os.path.join(out_dir, "gemm_4x4_vectors.txt")
    export_gemm_suite(out_file_4, num_tests=25, arr_m=4, arr_n=4, seed=123)

    # Export 16x16 suite
    out_file_16 = os.path.join(out_dir, "gemm_16x16_vectors.txt")
    export_parameterized_gemm_suite(out_file_16, num_tests=30, arr_m=16, arr_n=16, seed=456)
    print(f"Exported 30 GEMM 16x16 test cases -> {out_file_16}")
