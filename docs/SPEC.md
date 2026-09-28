# INT8 NPU — Design Specification

A quantized matrix-multiply accelerator for CNN inference, in SystemVerilog,
taken to synthesis and static timing signoff on SkyWater 130 nm.

**Author:** Dhruv Wagh
**Started:** 2026-09-28
**Scope:** RTL + verification + synthesis + STA. No place-and-route.

---

## 1. Goal and success criteria

Build a systolic-array accelerator that runs a quantized MNIST CNN **bit-exactly
against a PyTorch/NumPy reference**, then characterise it honestly.

The project is done when all of these hold:

1. Full network runs in RTL simulation, output matches the golden model **exactly**
   (not "within tolerance" — integer arithmetic, so exact or broken).
2. Constrained-random GEMM testing passes across randomised M/N/K with functional
   coverage closed on tile geometries and backpressure.
3. Yosys synthesis to sky130 HD completes with a reported gate/area breakdown,
   SRAM reported separately as macros.
4. OpenSTA closes setup and hold at a stated clock, with the critical path
   identified and root-caused.
5. A **per-layer utilization and roofline analysis** explaining where cycles go.

Criterion 5 is the deliverable that differentiates this from a tutorial project.

---

## 2. Design decisions and why

| Decision | Choice | Reason |
|---|---|---|
| Dataflow | **Output-stationary** | Accumulators are 32-bit, inputs are 8-bit. OS is the only dataflow that never moves a partial sum — each PE accumulates in place for the entire reduction and writes out exactly one value. WS would spend the whole bandwidth budget shuttling 4x-wide partials. |
| Array topology | **True systolic** (operands hop PE-to-PE) | Short local wires, no 16-way fanout. A broadcast array is simpler to control but the fanout wrecks timing at this width. |
| Array size | **16x16, parameterized** | Bring up at 4x4, verify at 8x8, characterise at 16x16. `ARR_M`/`ARR_N` are parameters throughout. |
| Precision | INT8 x INT8 -> INT32 | Standard quantized inference. 32-bit accum is oversized for MNIST (max K=784 needs 25 bits) but keeps headroom; `ACC_W` is a parameter and a real area lever. |
| Buffers | Blackboxed SRAM wrapper | Prevents flop inference from destroying area numbers. See section 6. |
| Host interface | AXI4-Lite slave for CSRs | Industry standard, reusable, and keeps SoC integration possible later without committing to it now. |
| Data port | Wide (128-bit), separate from CSR bus | See section 3 — this is the crux of the design. |

---

## 3. The bandwidth problem (read this before anything else)

A 16x16 output-stationary array consumes, every cycle:

- 16 activation bytes (one per row)
- 16 weight bytes (one per column)

= **32 bytes/cycle**. At 50 MHz that is 1.6 GB/s.

A 32-bit bus supplies 4 B/cycle. **The array would idle 87% of the time.**

Arithmetic intensity of a single MxNxK tile with M=N=16:

```
MACs fetched per byte = (16*16*K) / (32*K) = 8 MACs/byte
```

Note this is independent of K — deepening the reduction does not improve it,
because both operands stream. Intensity only improves by **reusing a tile across
multiple output tiles**.

### Mitigations, both of which are built in

1. **Weight-tile residency.** Hold one weight tile (M_TILE x K_TILE) in the
   buffer and sweep it across all N tiles of the activation matrix. Weights are
   fetched once per (M,K) tile instead of once per output tile. For conv2
   (N=196, so 13 N-tiles) this cuts weight traffic ~13x.
2. **Wide internal port.** The tensor memory port is `MEM_DW = 128` bits
   (16 B/cycle), not 32. Only the CSR bus is narrow.

### The experiment

`MEM_DW` is a parameter. Run the full network at `MEM_DW` = 32, 64, 128, 256 and
plot achieved GOPS against the bandwidth roofline. This turns the bottleneck into
a measured result rather than an assertion, and it is the single most valuable
figure in the final writeup.

---

## 4. Target workload

Quantized MNIST CNN, INT8 weights and activations, per-channel weight scales.

| Layer | Shape | im2col GEMM (M x N x K) | MACs |
|---|---|---|---|
| conv1 | 1x28x28 -> 8x28x28, 3x3 pad 1 | 8 x 784 x 9 | 56,448 |
| pool1 | maxpool 2x2 -> 8x14x14 | (software, v1) | — |
| conv2 | 8x14x14 -> 16x14x14, 3x3 pad 1 | 16 x 196 x 72 | 225,792 |
| pool2 | maxpool 2x2 -> 16x7x7 | (software, v1) | — |
| fc | 784 -> 10 | 10 x 1 x 784 | 7,840 |
| | | **Total** | **~290K MACs** |

Ideal time at 256 MAC/cycle: ~1,133 cycles, ~23 us at 50 MHz. Small enough to
simulate the whole network in RTL, which is what makes bit-exact validation
practical.

### Predicted utilization (verify this in RTL — it is the headline result)

| Layer | Why | Expected utilization |
|---|---|---|
| conv1 | M=8 fills only half the 16 rows; K=9 is shorter than the pipeline fill | ~50%, worse with fill/drain overhead |
| conv2 | M=16 exact fit, K=72 amortises fill | ~94% |
| fc | **N=1** — one active column out of 16 | ~4% |

The FC result is not a bug. A batch-1 fully-connected layer is a matrix-vector
product, which cannot fill a 2D array. This is exactly why inference accelerators
batch requests, and explaining it with your own measured numbers is a strong
interview answer.

**Scale-up path:** once MNIST is bit-exact, add a ResNet-style 3x3/1x1 block to
show the compiler and array generalise.

---

## 5. Module hierarchy

```
npu_top.sv
├── npu_pkg.sv            types, parameters, descriptor struct
├── npu_csr.sv            AXI4-Lite slave: config, control, status, perf counters
├── npu_sequencer.sv      FSM: M/N/K tile loops, buffer swap, array control
├── npu_fetch.sv          wide reads from tensor memory into ping-pong buffers
├── npu_buf_act.sv        ping-pong activation buffer   ─┐
├── npu_buf_wgt.sv        ping-pong weight buffer       ─┴─ wrap npu_sram.sv
├── npu_skew.sv           triangular delay registers for systolic input timing
├── npu_array.sv          ARR_M x ARR_N PE grid
│   └── npu_pe.sv         INT8 MAC + ACC_W accumulator + pass-through registers
├── npu_drain.sv          column-by-column accumulator readout
├── npu_epilogue.sv       bias add + requantize + ReLU + saturate to INT8
├── npu_obuf.sv           output staging, writeback to tensor memory
└── npu_sram.sv           behavioral for sim / blackbox for synth
```

Plus, outside the DUT:

```
tb/model/                 NumPy golden reference (bit-exact)
compiler/                 ONNX -> quantize -> tile -> descriptor stream
```

### npu_pe.sv — the leaf cell

```systemverilog
// Output-stationary PE. Operands hop through; the accumulator stays.
module npu_pe #(parameter int ACC_W = 32) (
  input  logic                clk, rst_n,
  input  logic                en,        // advance this cycle
  input  logic                clr,       // clear accumulator (start of new output tile)
  input  logic signed [7:0]   a_in,      // from west
  input  logic signed [7:0]   b_in,      // from north
  output logic signed [7:0]   a_out,     // to east   (registered a_in)
  output logic signed [7:0]   b_out,     // to south  (registered b_in)
  output logic signed [ACC_W-1:0] acc
);
```

Each PE is one signed 8x8 multiply, one ACC_W add, and three registers. The
`a_out`/`b_out` pass-through registers are what make it systolic: data arrives at
PE(i,j) exactly i+j cycles after entering the array, which is why the inputs must
be skewed on the way in.

- **Fill latency:** `ARR_M + ARR_N - 2` = 30 cycles for 16x16
- **Steady state:** 256 MACs/cycle
- **Drain:** `ARR_N` cycles, overlappable with the next tile's fill

---

## 6. Memory sizing

With `M_TILE = N_TILE = 16` and `K_TILE = 128`:

| Buffer | Size | Notes |
|---|---|---|
| Weight tile | 16 x 128 B = 2 KB | x2 ping-pong = **4 KB** |
| Activation tile | 128 x 16 B = 2 KB | x2 ping-pong = **4 KB** |
| Output staging | 16 x 16 B = 256 B | x2 = **512 B** |
| Bias | 16 x 32b = 64 B | registers, not SRAM |
| Accumulators | 256 x 32b = **8,192 flops** | inside the PEs |

**Total SRAM: ~8.5 KB. Total array flops: ~8.2K.**

`K > K_TILE` (the FC layer, K=784) is handled by keeping the accumulators live
across successive K tiles and only draining once the reduction completes. This is
free with output-stationary and is the main reason to prefer it.

### The flop-inference trap

Do **not** write buffers as bare arrays. 8.5 KB inferred as flops is ~69,000
flip-flops, roughly 1.5 mm2 on sky130 HD — four times the compute array, and it
would make every area number in the report meaningless.

Instead, `npu_sram.sv` is a wrapper:

```systemverilog
module npu_sram #(parameter int DW = 128, parameter int WORDS = 128) (...);
`ifdef SYNTHESIS
  // empty — blackboxed, area accounted separately from a macro datasheet
`else
  logic [DW-1:0] mem [WORDS];   // behavioral model for simulation
`endif
endmodule
```

Report area as **"X um2 of logic + N SRAM macros of size Y"**. That is how a real
design is reported, and it is defensible in an interview in a way that a single
inflated number is not.

**Area lever to note:** `ACC_W` 32 -> 24 removes 2,048 flops (~25% of array
registers) and is safe for K <= 128. Worth measuring and mentioning.

---

## 7. Register map (AXI4-Lite slave, byte offsets)

| Offset | Name | Access | Description |
|---|---|---|---|
| 0x00 | `CTRL` | W | bit0 `start`, bit1 `abort`, bit2 `perf_clr` |
| 0x04 | `STATUS` | R | bit0 `busy`, bit1 `done`, bit2 `err` |
| 0x08 | `DIM_M` | RW | GEMM M (output channels) |
| 0x0C | `DIM_N` | RW | GEMM N (output spatial positions) |
| 0x10 | `DIM_K` | RW | GEMM K (reduction depth) |
| 0x14 | `A_BASE` | RW | weight matrix base address |
| 0x18 | `B_BASE` | RW | activation (im2col) matrix base address |
| 0x1C | `C_BASE` | RW | output base address |
| 0x20 | `BIAS_BASE` | RW | per-output-channel bias base |
| 0x24 | `RQ_MULT` | RW | requantize multiplier (int32) |
| 0x28 | `RQ_SHIFT` | RW | requantize right shift |
| 0x2C | `FLAGS` | RW | bit0 `relu_en`, bit1 `bias_en`, bit2 `signed_act` |
| 0x30 | `PERF_CYCLES` | R | total cycles for the layer |
| 0x34 | `PERF_MACS` | R | MAC-cycles actually issued |
| 0x38 | `PERF_STALL` | R | cycles stalled on buffer fill |

`PERF_MACS / (PERF_CYCLES * ARR_M * ARR_N)` gives per-layer utilization directly
in hardware. Build these counters in from day one — they are how section 4's
table gets measured rather than estimated.

---

## 8. Requantization (get this exactly right or nothing is bit-exact)

After the reduction completes, each INT32 accumulator becomes an INT8:

```
acc32              : signed 32-bit accumulator
+ bias[m]          : signed 32-bit, if bias_en
* RQ_MULT          : signed 32x32 -> 64-bit
>> RQ_SHIFT        : rounding right shift (round half away from zero)
ReLU               : max(0, x), if relu_en
saturate           : clamp to [-128, 127]
```

**Bit-exactness lives or dies on the rounding mode.** TFLite and PyTorch use a
specific rounding-shift convention; pick one reference and match it in both the
RTL and the NumPy golden model. Write the golden model *first*, generate vectors
from it, and make the RTL match — not the other way around.

---

## 9. Compiler IR

Python, `compiler/`. Four stages:

```
ONNX model
  │  1. import + fold BatchNorm into conv weights
  ▼
Graph (layer list)
  │  2. quantize: per-channel INT8 weights, per-tensor INT8 activations,
  │               compute RQ_MULT / RQ_SHIFT per layer
  ▼
QuantGraph
  │  3. lower: conv -> im2col -> GEMM(M,N,K); emit layout + strides
  ▼
GemmOp list
  │  4. tile to array geometry, order loops for weight residency
  ▼
Descriptor stream (binary, loaded by the TB)
```

### Core IR types

```python
@dataclass
class GemmOp:                 # one layer, after im2col lowering
    name: str
    M: int; N: int; K: int    # out_ch, out_spatial, in_ch*kh*kw
    a_base: int               # weights
    b_base: int               # im2col activations
    c_base: int               # outputs
    bias_base: int
    rq_mult: int              # int32
    rq_shift: int
    relu: bool

@dataclass
class Tile:                   # one array-sized work unit
    op: str                   # 'gemm'
    m0: int; n0: int; k0: int # tile origin
    m_len: int; n_len: int; k_len: int
    first_k: bool             # clear accumulators
    last_k:  bool             # run epilogue + drain
    # address/stride fields resolved from the parent GemmOp
```

`first_k` / `last_k` are what let K exceed `K_TILE`: accumulators clear on the
first K tile, drain on the last, and simply keep accumulating in between.

**Loop order matters.** Iterate `M -> K -> N` so a weight tile stays resident
across the whole N sweep. Iterating N outermost would refetch weights every tile
and cost you the 13x traffic reduction on conv2.

---

## 10. Verification plan

Three layers, built in this order:

**1. Golden model first (`tb/model/`).** NumPy, bit-exact, including the exact
requantize rounding. Everything else is validated against this. Writing it first
means the RTL has a target instead of a guess.

**2. Unit testbenches (`tb/unit/`).** One per module. `npu_pe` against a scalar
MAC; `npu_array` against `numpy.matmul` on random INT8 with random M/N/K;
`npu_epilogue` swept across the full accumulator range including saturation
boundaries at exactly -128 and +127.

**3. SystemVerilog verification environment.** This is a hiring signal in its own
right, so do it properly:

- **Constrained-random stimulus** — random M/N/K including degenerate cases
  (K=1, N=1, M < ARR_M, dims one above and one below tile boundaries)
- **Functional coverage** — cover bins on tile geometry, partial-tile edges,
  K-tiling depth, buffer swap under backpressure, accumulator saturation hit
- **SVA assertions** — AXI-Lite protocol compliance, no accumulator drain while
  `!last_k`, no buffer swap mid-tile, `busy`/`done` mutual exclusion
- **End-to-end** — full MNIST network, bit-exact vs the golden model, across the
  full test set if simulation time allows

Cocotb is a reasonable alternative to UVM here and you already know it; the point
is coverage-driven randomised verification, not the specific framework.

---

## 11. Milestones

Roughly 8 weeks part-time. Each has a hard exit criterion.

| # | Milestone | Exit criterion | Est. |
|---|---|---|---|
| 1 | `npu_pkg`, `npu_pe` + unit TB, NumPy golden MAC | PE matches scalar MAC over random vectors | 3 days |
| 2 | `npu_array` 4x4, skew registers | 4x4 GEMM matches `numpy.matmul`, fill/drain latency matches hand calc | 1 wk |
| 3 | Scale to 16x16, parameterized | Random M/N/K <= tile size passes | 3 days |
| 4 | `npu_sram`, ping-pong buffers, `npu_fetch` | Buffer swap overlaps fetch with compute, zero bubbles in steady state | 1 wk |
| 5 | `npu_sequencer` — full M/N/K tiling | Arbitrary M/N/K including K > K_TILE and partial edge tiles | 1 wk |
| 6 | `npu_epilogue` + `npu_obuf` | Bias/requant/ReLU/saturate bit-exact vs golden across full range | 4 days |
| 7 | `npu_csr` AXI-Lite + perf counters | CPU-less TB programs a layer and reads back correct utilization | 4 days |
| 8 | Compiler: ONNX -> quantize -> tile -> descriptors | MNIST conv1 runs end-to-end, bit-exact | 1 wk |
| 9 | Full network + SV coverage closure | All layers bit-exact; functional coverage closed | 1 wk |
| 10 | Yosys synth + OpenSTA | Timing closed at stated clock, critical path root-caused, area broken out | 4 days |
| 11 | `MEM_DW` sweep + roofline + utilization writeup | The two headline figures produced | 4 days |

Milestones 1-3 are the fun part and go fast. **Milestone 5 is where the real
difficulty is** — tiling with partial edge tiles and K-splitting is where most
accelerator projects break. Budget more time there than feels necessary.

---

## 12. Open questions

1. Max pooling: software in v1, or a small hardware pooling unit in the epilogue?
2. Does im2col happen in the compiler (simple, more memory traffic) or in the
   fetch unit as an address-generation trick (harder, much less traffic)?
   Recommend compiler-side for v1, then measure the traffic cost.
3. Target clock for STA — pick a number and design to it rather than discovering
   it. 50 MHz is a reasonable sky130 target for this structure.
4. Cocotb or a UVM-lite SystemVerilog env for milestone 9?
