# INT8 NPU — Engineering Log

Detailed chronological record of design decisions, microarchitectural trade-offs, implementation steps, and verification results.

---

## 2026-09-28 — Milestone 1: Leaf Processing Element (`npu_pe`) & Golden Reference

### 1. Context & Objectives
- **Target:** Establish the fundamental computational leaf cell of the systolic array, define the architecture-wide parameters/types package, and build a bit-exact NumPy reference model.
- **Exit Criterion (from SPEC.md):** `npu_pkg`, `npu_pe` + unit testbench, NumPy golden MAC; PE matches scalar MAC over random vectors and synthesis confirms zero inferred latches.

### 2. Microarchitectural Decisions & Rationale

#### A. Arithmetic Precision & Overflow Sizing
- **INT8 $\times$ INT8 $\to$ INT16:** Standard two's complement multiplication. Inputs span $[-128, +127]$. The extreme negative multiplication $(-128) \times (-128) = +16384$ ($0\text{x}4000$) fits strictly within a signed 16-bit word ($[-32768, +32767]$).
- **Accumulator Width (`ACC_W = 32`):** Sized to 32 bits to prevent overflow across long GEMM reductions. For MNIST Conv2 ($K=72$), maximum accumulation magnitude is $72 \times 16384 \approx 1.18 \times 10^6$ (requires 21 bits). For FC ($K=784$), worst-case magnitude is $784 \times 16384 \approx 1.28 \times 10^7$ (requires 24 bits). 32 bits provides ample headroom and standard bus alignment.

#### B. Output-Stationary Dataflow Semantics
- In an output-stationary array, accumulators remain stationary inside each PE throughout the tile reduction.
- Operands stream systolically:
  - Activations $A[i]$ hop from West $\to$ East via registered `a_out <= a_in`.
  - Weights $B[j]$ hop from North $\to$ South via registered `b_out <= b_in`.
- Data hops one PE per cycle, requiring an input skew triangle so that operand pairs $A_{i,k}$ and $B_{k,j}$ arrive at $\text{PE}(i,j)$ simultaneously at cycle $t_0 + i + j$.

#### C. Control Signals (`en` and `clr`)
- **`en` (Advance Enable):** Stalls pipeline registers (`a_out`, `b_out`) and pauses accumulation without corrupting state when the memory interface stalls.
- **`clr` (Accumulator Clear):**
  - Designed with dual-mode flexibility:
    1. If `clr && en`: Atomically initializes accumulator to the current product ($0 + a_{in} \times b_{in}$). This enables back-to-back zero-bubble tile switching.
    2. If `clr && !en`: Resets accumulator to 0 during an idle or setup cycle.

### 3. Implementation Steps

1. **NumPy Golden Reference (`model/golden_mac.py`):**
   - Created `GoldenPE` class modeling exact two's complement behavior, sign extension, and register states.
   - Built `generate_pe_vectors()` producing 1,000 diverse test vectors covering:
     - Asynchronous reset assertions
     - Extreme corners: $(-128) \times (-128)$, $(-128) \times 127$, $127 \times 127$, zero multipliers
     - Accumulator clearing (`clr=1`) under both stalled (`en=0`) and active (`en=1`) conditions
     - Stall cycles (`en=0`) to verify memory retention
     - Deep accumulation runs (200 cycles)
     - Constrained pseudorandom stimulus

2. **SystemVerilog Package (`src/npu_pkg.sv`):**
   - Defined default array dimensions (`ARR_M=16, ARR_N=16`), operand widths (`DATA_W=8, ACC_W=32, MEM_DW=128`).
   - Mapped AXI4-Lite CSR offsets (`0x00` through `0x38`) according to SPEC Section 7.
   - Defined hardware performance counter registers (`PERF_CYCLES`, `PERF_MACS`, `PERF_STALL`) for roofline profiling.
   - Defined tile descriptor struct `tile_desc_t` for sequencer-to-compute dispatch.

3. **Leaf Processing Element (`src/pe/npu_pe.sv`):**
   - Implemented `npu_pe` module parameterized by `ACC_W`.
   - Used explicit sign extension `prod_ext = {{ (ACC_W-16){prod[15]} }, prod}` to ensure clean synthesis without tool ambiguity.
   - Clean asynchronous active-low reset `rst_n`.

4. **Unit Testbench (`tb/unit/tb_npu_pe.sv`):**
   - Applied clock (100 MHz / 10 ns period).
   - Applied vectors from `tb/vectors/pe_vectors.hex` on `negedge clk` and sampled on `posedge clk + 1ns`.
   - Included assertion checks on all output ports (`a_out`, `b_out`, `acc`).

5. **Automated Test Runners (`scripts/run_pe_test.py`, `scripts/sim_pe.sh`):**
   - Automated vector generation, compilation with Icarus Verilog (`iverilog`), Verilator 5 (`verilator`), and synthesis gate audit with Yosys.

### 4. Verification Results & Audit
- **Simulation:**
  - `iverilog`: 1000/1000 vectors passed (0 errors).
  - `verilator 5.050`: 1000/1000 vectors passed (0 errors).
- **Yosys Synthesis Check:**
  - Cell count: Exactly **48 DFFs** inferred (`$_DFFE_PN0P_`):
    - 8 bits for `a_out`
    - 8 bits for `b_out`
    - 32 bits for `acc`
  - Combinational logic mapped to 740 gate cells.
  - 0 latches, 0 multi-driven nets, 0 combinational loops.

---

## 2026-09-28 — Milestone 2: 4x4 Systolic Array (`npu_array`), Skew Registers (`npu_skew`), & Drain Unit (`npu_drain`)

### 1. Context & Objectives
- **Target:** Scale compute from single leaf PE to a 2D systolic array grid with triangular skew timing registers and accumulator drain logic.
- **Exit Criterion (from SPEC.md):** `npu_array` 4x4, skew registers; 4x4 GEMM matches `numpy.matmul` bit-exactly across varying $K$ (including backpressure stalls), and fill/drain latency matches hand calculation.

### 2. Microarchitectural Decisions & Mathematical Derivations

#### A. Systolic Input Skew Triangle (`npu_skew.sv`)
- **Activation Skew:**
  - Operands hop horizontally from West to East with 1-cycle latency per PE.
  - An operand pair $(A_{i,k}, B_{k,j})$ must arrive at $\text{PE}(i, j)$ at cycle $t_0 + i + j$.
  - Therefore, row $i \in [0, \text{ARR\_M}-1]$ requires an input delay of exactly $i$ clock cycles before reaching $\text{PE}(i, 0)$.
  - Implemented as triangular shift register pipelines of depth $i$ enabled by clock enable `en`.
- **Weight Skew:**
  - Operands hop vertically from North to South with 1-cycle latency per PE.
  - Column $j \in [0, \text{ARR\_N}-1]$ requires an input delay of exactly $j$ clock cycles before reaching $\text{PE}(0, j)$.
  - Implemented as triangular shift register pipelines of depth $j$ enabled by clock enable `en`.
- **Unified Systolic Clear (`clr_skewed`):**
  - $\text{PE}(i, j)$ computes its very first product ($k=0$) at cycle $t_0 + i + j$. Its accumulator must clear at this exact cycle.
  - Instead of replicating independent shift registers for all $M \times N$ cells, notice that $D = i + j$ takes only $\text{ARR\_M} + \text{ARR\_N} - 1$ unique values ($0 \le D \le \text{ARR\_M} + \text{ARR\_N} - 2$).
  - Implemented as a single, unified 1-bit shift register pipeline `clr_pipe` of depth $\text{ARR\_M} + \text{ARR\_N} - 2$.
  - For $4 \times 4$, this reduces clear registers from 16 to **6 flip-flops**; for $16 \times 16$, from 256 to **30 flip-flops** (~88% register savings).

#### B. Hand Calculation of Array Latencies
- **Fill Latency:**
  - Defined as the elapsed time from injecting the first operands ($k=0$) at array inputs to the arrival of their product at the furthest bottom-right cell $\text{PE}(\text{ARR\_M}-1, \text{ARR\_N}-1)$.
  - Formula:
    $$\text{Latency}_{\text{fill}} = (\text{ARR\_M}-1) + (\text{ARR\_N}-1) = \text{ARR\_M} + \text{ARR\_N} - 2$$
  - For $4 \times 4$: $4 + 4 - 2 = \mathbf{6\text{ cycles}}$.
  - For $16 \times 16$: $16 + 16 - 2 = \mathbf{30\text{ cycles}}$.
- **Total Compute Latency:**
  - For reduction depth $K$:
    $$\text{Cycles}_{\text{compute}} = K + (\text{ARR\_M} + \text{ARR\_N} - 2)$$
- **Drain Latency (`npu_drain.sv`):**
  - Defined as the number of cycles to read out all accumulators column-by-column.
  - With $N$ columns and 1 column drained per clock cycle:
    $$\text{Latency}_{\text{drain}} = \text{ARR\_N} = \mathbf{4\text{ cycles for 4x4}}\text{ (16 cycles for 16x16)}$$
  - Snapshot shadow registers allow the array to immediately accept the next tile's operands, overlapping drain with next-tile fill.

#### C. Cross-Tool Synthesizable Port Architecture
- **Challenge:** Icarus Verilog, Verilator, and Yosys handle multi-dimensional SystemVerilog port declarations differently. Unpacked array ports (e.g. `input logic a [M][N]`) fail in Yosys's default Verilog parser, while multi-dimensional packed indexing on variables (e.g. `acc[m][col_cnt]`) fails in Icarus Verilog's elaborator.
- **Solution:** Standardized inter-module boundaries to 1D flattened packed buses (e.g. `[ARR_M*ARR_N*ACC_W-1:0]`) using standard Verilog-2001 `+:` part-select indexing. Unpacked 2D representations are maintained strictly inside module interiors. This guarantees 100% interoperability across Yosys, Verilator, Icarus, and commercial ASIC synthesis flows.

### 3. Implementation Steps

1. **Cycle-Accurate & Golden Model (`model/golden_gemm.py`):**
   - Implemented `int8_gemm_golden` using `numpy.matmul` with two's complement `ACC_W` saturation wrapping.
   - Built `CycleAccurateSystolicArray` modeling every internal skew register and PE hop cycle-by-cycle.
   - Implemented `export_gemm_suite()` generating 25 diverse test vectors ($K \in [1, 2, 4, 8, 16]$, identity, extreme signed corners $[-128, +127]$).

2. **Skew Module (`src/array/npu_skew.sv`):**
   - Triangular shift registers for row activations and column weights with parameterizable dimensions.
   - Unified `clr_pipe` routing `clr_pipe[i + j]` to $\text{PE}(i, j)$.

3. **Systolic Array (`src/array/npu_array.sv`):**
   - 2D grid instantiating $\text{ARR\_M} \times \text{ARR\_N}$ instances of `npu_pe`.
   - West-to-East horizontal registered forwarding (`a_h`), North-to-South vertical registered forwarding (`b_v`).
   - All 32-bit accumulators routed out in parallel.

4. **Column Drain Unit (`src/array/npu_drain.sv`):**
   - Captures accumulator snapshot at `drain_start`.
   - Sequential counter sweeps column index $0 \to \text{ARR\_N}-1$.
   - Drives $M \times \text{ACC\_W}$ vector per cycle, asserting `drain_done` on the exact $N$-th cycle.

5. **Unit Testbench (`tb/unit/tb_npu_array_4x4.sv`):**
   - Measures and asserts fill latency ($6$ cycles) and drain latency ($4$ cycles).
   - Verifies 25 GEMM test cases covering varying $K$, extreme bounds, and back-to-back zero-bubble tiles.
   - Injects random backpressure stalls (`en=0`) with noisy inputs to verify state retention.

6. **Automated Test Runners (`scripts/run_array_4x4_test.py`, `scripts/sim_array_4x4.sh`):**
   - Re-generates test vectors, runs Icarus Verilog and Verilator 5 simulations, and audits Yosys synthesis netlists.

### 4. Verification Results & Audit

- **Simulation Results:**
  ```text
  [CHECK] Testing Fill Latency: hand calc says PE(3,3) activates at cycle 6...
  [RESULT] Fill latency measured: 6 cycles (Expected: 6)
  [PASS] Hand-calculated fill latency matches EXACTLY (6 cycles).

  [CHECK] Testing Drain Latency: hand calc says drain takes exactly ARR_N = 4 cycles...
  [RESULT] Drain latency measured: 4 cycles (Expected: 4)
  [PASS] Hand-calculated drain latency matches EXACTLY (4 cycles).

  [INFO] Loading GEMM test suite from tb/vectors/gemm_4x4_vectors.txt...
  [PASS] Test 0  (M=4, N=4, K=4)  matches NumPy golden exactly.
  [PASS] Test 1  (M=4, N=4, K=4)  matches NumPy golden exactly.
  ...
  [PASS] Test 24 (M=4, N=4, K=2)  matches NumPy golden exactly.
  Total GEMM tests evaluated : 25
  Failed tests               : 0
  RESULT: *** ALL TESTS PASSED *** (Bit-exact match to NumPy reference)
  ```
- **Simulators Tested:**
  - `iverilog`: 25/25 passed (0 errors)
  - `verilator 5.050`: 25/25 passed (0 errors)
- **Yosys Synthesis Check (ARR_M=4, ARR_N=4):**
  - **DFF Count:** Exactly **768 DFFs** inferred for array compute logic ($16\text{ PEs} \times 48\text{ DFFs/PE}$).
  - **Structure:** 16 instantiated `npu_pe` leaf modules.
  - **Sanity:** 0 latches, 0 multi-driven nets, 0 combinational loops.

---

## 2026-09-28 — Milestone 3: Scale to 16x16 Systolic Array & Parameterized Sub-Tile Verification

### 1. Context & Objectives
- **Target:** Scale the 2D compute array to full 16×16 dimensions (256 PEs), verify fill/drain latency at full scale, and rigorously prove bit-exact correctness across randomized full tiles and arbitrary partial/sub-tiles ($M \le 16, N \le 16, K \le 64$).
- **Exit Criterion (from SPEC.md):** Scale to 16x16, parameterized; random $M/N/K \le \text{tile size}$ passes bit-exactly against NumPy model; fill latency (30 cycles) and drain latency (16 cycles) verified; VCD waveforms generated.

### 2. Microarchitectural Decisions & Analysis

#### A. 16×16 Array Scaling & Register Budget
- **Compute Grid:** $16 \times 16 = 256$ processing elements.
  - Flop breakdown per PE: 8 (`a_out`) + 8 (`b_out`) + 32 (`acc`) = 48 DFFs.
  - Total array flops: $256 \times 48 = \mathbf{12,288\text{ DFFs}}$ (matches SPEC Section 6).
- **Latency Scaling:**
  - **Fill Latency:**
    $$\text{ARR\_M} + \text{ARR\_N} - 2 = 16 + 16 - 2 = \mathbf{30\text{ clock cycles}}$$
    This is the exact number of cycles for the first operand pair $(A_{15,0}, B_{0,15})$ to reach the bottom-right corner cell $\text{PE}(15,15)$.
  - **Drain Latency:**
    $$\text{ARR\_N} = \mathbf{16\text{ clock cycles}}$$
    One column of 16 accumulators ($16 \times 32\text{ bits} = 512\text{ bits}$) drained per clock cycle into shadow buffers.
  - **Peak Throughput:** 256 MACs/cycle (at 50 MHz = 12.8 GMAC/s or 25.6 GOP/s).

#### B. Sub-Tile & Degenerate Geometry Handling
- Neural network layers frequently exhibit dimensions not aligned to the $16 \times 16$ tile boundary:
  1. **MNIST FC Layer ($N=1$):** A matrix-vector product ($M=10, N=1, K=784$). Only 1 active column out of 16 ($\approx 6.25\%$ spatial utilization).
  2. **Conv1 Layer ($M=8$):** Only 8 output channels ($M=8 < \text{ARR\_M}$), filling only half the PE rows.
  3. **Edge Tiles:** Output feature map edges produce spatial tiles with $N < 16$.
- **Hardware Zero-Padding Strategy:**
  - Inactive activation rows ($m \ge M_{\text{active}}$) are driven with 0.
  - Inactive weight columns ($n \ge N_{\text{active}}$) are driven with 0.
  - Because $0 \times x = 0$, unallocated PEs accumulate zero without corrupting neighboring cells.
  - Valid output matrices $C[0..M_{\text{act}}-1, 0..N_{\text{act}}-1]$ emerge bit-exact from the array with zero hardware mode switches or pipeline reconfigurations required.

### 3. Implementation Steps

1. **Golden Model Suite Expansion (`model/golden_gemm.py`):**
   - Implemented `export_parameterized_gemm_suite()` generating 30 diverse test cases:
     - Test 0: Full 16×16 identity matrix
     - Test 1: Full 16×16 extreme values ($+127 \times -128$)
     - Test 2: Degenerate $N=1$ vector product ($10 \times 1 \times 16$)
     - Test 3: Degenerate $M=1$ single-channel output ($1 \times 16 \times 12$)
     - Test 4: Degenerate $K=1$ minimal reduction ($16 \times 16 \times 1$)
     - Test 5: Conv1-like shape ($M=8, N=16, K=9$)
     - Tests 6–29: Constrained-random sub-tiles ($M \in [1, 16], N \in [1, 16], K \in [1, 64]$) with backpressure stall bubble injection.

2. **16×16 Verification Environment (`tb/unit/tb_npu_array_16x16.sv`):**
   - Instantiated `npu_skew`, `npu_array`, and `npu_drain` at `ARR_M=16, ARR_N=16`.
   - Measured and asserted fill latency (30 cycles) and drain latency (16 cycles).
   - Validated all 30 test cases against NumPy golden outputs.
   - Dumped full waveform database to `sim_output/tb_npu_array_16x16.vcd` (31 MB).

3. **Automation & Waveform Viewing (`scripts/run_array_16x16_test.py`, `scripts/sim_array_16x16.sh`, `scripts/view_wave.sh`):**
   - Automated dual-simulator execution (`iverilog` and `verilator`) plus Yosys gate-level synthesis audit.
   - Built `scripts/view_wave.sh` for one-click inspection in GTKWave.

### 4. Verification Results & Synthesis Audit

- **Simulation Results:**
  ```text
  [CHECK] Testing Fill Latency: hand calc says PE(15,15) activates at cycle 30...
  [RESULT] 16x16 Fill latency measured: 30 cycles (Expected: 30)
  [PASS] Hand-calculated 16x16 fill latency matches EXACTLY (30 cycles).

  [CHECK] Testing Drain Latency: hand calc says drain takes exactly ARR_N = 16 cycles...
  [RESULT] 16x16 Drain latency measured: 16 cycles (Expected: 16)
  [PASS] Hand-calculated 16x16 drain latency matches EXACTLY (16 cycles).

  [INFO] Loading 16x16 GEMM test suite from tb/vectors/gemm_16x16_vectors.txt...
  [PASS] Test 0  (Active: M=16, N=16, K=16) bit-exact match vs NumPy.
  [PASS] Test 1  (Active: M=16, N=16, K=8)  bit-exact match vs NumPy.
  [PASS] Test 2  (Active: M=10, N=1,  K=16) bit-exact match vs NumPy. (FC layer)
  [PASS] Test 3  (Active: M=1,  N=16, K=12) bit-exact match vs NumPy.
  [PASS] Test 4  (Active: M=16, N=16, K=1)  bit-exact match vs NumPy.
  [PASS] Test 5  (Active: M=8,  N=16, K=9)  bit-exact match vs NumPy. (Conv1 shape)
  ...
  [PASS] Test 29 (Active: M=16, N=4,  K=3)  bit-exact match vs NumPy.
  Total GEMM tests evaluated : 30
  Failed tests               : 0
  RESULT: *** ALL 16x16 TESTS PASSED *** (Bit-exact match to NumPy)
  ```
- **Simulators Tested:**
  - `iverilog`: 30/30 passed (0 errors)
  - `verilator 5.050`: 30/30 passed (0 errors)
- **Yosys Synthesis Check (16×16 Default):**
  - **DFF Count:** Exactly **12,288 DFFs** inferred for array compute logic ($256\text{ PEs} \times 48\text{ DFFs/PE}$).
  - **Submodules:** 256 instantiated `npu_pe` leaf modules.
  - **Sanity:** 0 latches, 0 multi-driven nets, 0 combinational loops.
- **Waveforms Available:**
  - `sim_output/tb_npu_array_16x16.vcd` (31 MB)
  - `sim_output/tb_npu_array_4x4.vcd` (679 KB)
  - `sim_output/tb_npu_pe.vcd` (342 KB)

---
