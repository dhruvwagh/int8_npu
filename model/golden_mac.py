"""
Golden Python/NumPy Reference Model for INT8 NPU Processing Element (PE).

Matches npu_pe.sv bit-for-bit:
- INT8 x INT8 -> INT16 signed multiplication
- INT16 sign-extended to ACC_W (default 32-bit) accumulation
- Systolic operand forwarding (a_out = a_in, b_out = b_in) with 1-cycle latency when en=1
- Accumulator clear (clr) and enable (en) semantics
"""

from typing import Tuple, List, Dict, Any
import numpy as np


def int8_clamp(val: int) -> int:
    """Clamp integer to signed 8-bit range [-128, 127]."""
    if val > 127:
        return 127
    if val < -128:
        return -128
    return val


def to_signed(val: int, bits: int) -> int:
    """Convert integer to signed two's complement of given bit width."""
    val = val & ((1 << bits) - 1)
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


class GoldenPE:
    """Bit-exact golden reference model for npu_pe.sv."""

    def __init__(self, acc_w: int = 32):
        self.acc_w = acc_w
        self.acc_mask = (1 << acc_w) - 1
        self.reset()

    def reset(self) -> None:
        """Reset PE state to power-on/reset condition."""
        self.a_out = 0
        self.b_out = 0
        self.acc = 0

    def step(self, en: bool, clr: bool, a_in: int, b_in: int) -> Tuple[int, int, int]:
        """
        Advance one clock cycle.

        Args:
            en: Enable clock advance for registers and accumulation.
            clr: Clear accumulator (start of new output tile).
            a_in: Signed 8-bit activation input [-128, 127].
            b_in: Signed 8-bit weight input [-128, 127].

        Returns:
            Tuple of (a_out, b_out, acc) matching output ports after clock edge.
        """
        # Validate inputs are within INT8 range
        a = to_signed(a_in, 8)
        b = to_signed(b_in, 8)

        # Signed 8x8 multiply -> signed 16-bit product
        prod = a * b

        # Compute next accumulator
        if clr:
            if en:
                next_acc = prod
            else:
                next_acc = 0
        else:
            if en:
                next_acc = self.acc + prod
            else:
                next_acc = self.acc

        # Wrap to acc_w signed
        next_acc = to_signed(next_acc, self.acc_w)

        # Compute next systolic pass-through registers
        if en:
            next_a_out = a
            next_b_out = b
        else:
            next_a_out = self.a_out
            next_b_out = self.b_out

        # Commit state updates
        self.a_out = next_a_out
        self.b_out = next_b_out
        self.acc = next_acc

        return self.a_out, self.b_out, self.acc


def generate_pe_vectors(num_vectors: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a diverse suite of test vectors:
    1. Reset & initialization
    2. Zero inputs
    3. Small positive & negative values
    4. Extreme corner cases (-128 * -128, -128 * 127, 127 * 127)
    5. Accumulator clear while en=1 vs en=0
    6. Enable gating (stalls) with inputs changing
    7. Deep accumulation runs without clr
    8. Pseudorandom constrained vectors
    """
    rng = np.random.default_rng(seed)
    pe = GoldenPE(acc_w=32)
    vectors = []

    def record_step(rst_n: int, en: int, clr: int, a_in: int, b_in: int, comment: str = ""):
        if not rst_n:
            pe.reset()
            exp_a_out, exp_b_out, exp_acc = 0, 0, 0
        else:
            exp_a_out, exp_b_out, exp_acc = pe.step(bool(en), bool(clr), a_in, b_in)

        vectors.append({
            "rst_n": rst_n,
            "en": en,
            "clr": clr,
            "a_in": to_signed(a_in, 8),
            "b_in": to_signed(b_in, 8),
            "exp_a_out": exp_a_out,
            "exp_b_out": exp_b_out,
            "exp_acc": exp_acc,
            "comment": comment
        })

    # 1. Reset
    record_step(0, 0, 0, 0, 0, "Reset active")
    record_step(0, 1, 1, 10, -5, "Reset active with inputs")
    record_step(1, 0, 0, 0, 0, "Release reset, idle")

    # 2. Directed corner cases: Max bounds
    corners = [
        (127, 127, "Max positive product (127 * 127 = 16129)"),
        (-128, -128, "Max negative x max negative (-128 * -128 = 16384)"),
        (-128, 127, "Min x Max (-128 * 127 = -16256)"),
        (127, -128, "Max x Min (127 * -128 = -16256)"),
        (0, 127, "Zero x Max"),
        (-128, 0, "Min x Zero"),
        (1, -1, "1 * -1 = -1"),
        (-1, -1, "-1 * -1 = 1"),
    ]

    # Run corner cases with clear on first step
    for i, (a, b, desc) in enumerate(corners):
        clr = 1 if i == 0 else 0
        record_step(1, 1, clr, a, b, desc)

    # 3. Test enable gating: en=0 holds accumulator and pass-through registers
    record_step(1, 0, 0, 45, -30, "Stall cycle (en=0) - state should hold")
    record_step(1, 0, 0, 99, 99, "Stall cycle (en=0) with different inputs")

    # 4. Clear accumulator with en=0
    record_step(1, 0, 1, 0, 0, "Clear acc while stalled (en=0, clr=1)")
    record_step(1, 1, 0, 5, 5, "Resume compute: acc should be 0 + 25 = 25")

    # 5. Clear accumulator with en=1 (atomic clear and accumulate)
    record_step(1, 1, 1, 7, 7, "Atomic clear and accumulate: acc should be 49")

    # 6. Deep accumulation run (200 cycles of alternating positive/negative)
    for k in range(200):
        a = int(rng.integers(-128, 128))
        b = int(rng.integers(-128, 128))
        record_step(1, 1, 0, a, b, f"Deep accumulation step {k}")

    # 7. Randomized stimulus for remaining vectors
    remaining = num_vectors - len(vectors)
    for k in range(remaining):
        en = 1 if rng.random() > 0.1 else 0  # 90% en=1, 10% stall
        clr = 1 if rng.random() < 0.05 else 0  # 5% clr=1
        a = int(rng.integers(-128, 128))
        b = int(rng.integers(-128, 128))
        record_step(1, en, clr, a, b, f"Random test {k}")

    return vectors


def export_vectors_to_file(vectors: List[Dict[str, Any]], filepath: str) -> None:
    """
    Export vectors to a format easily parsed by $readmemh or $fscanf in Verilog.
    Columns (hex):
    rst_n en clr a_in(8b) b_in(8b) exp_a_out(8b) exp_b_out(8b) exp_acc(32b)
    """
    with open(filepath, "w") as f:
        f.write("# rst_n en clr a_in b_in exp_a_out exp_b_out exp_acc\n")
        for v in vectors:
            a_hex = f"{v['a_in'] & 0xFF:02x}"
            b_hex = f"{v['b_in'] & 0xFF:02x}"
            exp_a_hex = f"{v['exp_a_out'] & 0xFF:02x}"
            exp_b_hex = f"{v['exp_b_out'] & 0xFF:02x}"
            exp_acc_hex = f"{v['exp_acc'] & 0xFFFFFFFF:08x}"
            f.write(
                f"{v['rst_n']} {v['en']} {v['clr']} "
                f"{a_hex} {b_hex} {exp_a_hex} {exp_b_hex} {exp_acc_hex}\n"
            )


if __name__ == "__main__":
    import os
    import sys

    out_dir = os.path.join(os.path.dirname(__file__), "..", "tb", "vectors")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "pe_vectors.hex")

    count = 1000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])

    vecs = generate_pe_vectors(count)
    export_vectors_to_file(vecs, out_file)
    print(f"Generated {len(vecs)} golden test vectors -> {out_file}")
