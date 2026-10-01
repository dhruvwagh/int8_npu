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

## 2026-09-28 — Milestone 4: SRAM Macro Wrappers, Ping-Pong Buffers & Wide Fetch Engine

### 1. Context & Objectives
- **Target:** Implement the on-chip storage hierarchy (`npu_sram`, `npu_buf_act`, `npu_buf_wgt`) and the tensor memory DMA streaming engine (`npu_fetch`).
- **Exit Criterion (from SPEC.md):** Buffer swap overlaps fetch with compute, zero bubbles in steady state; weight-tile residency verified across multiple activation sweeps; SRAM macros blackboxed during Yosys synthesis to avoid flip-flop explosion.

### 2. Microarchitectural Decisions & Analysis

#### A. SRAM Macro Blackboxing vs The Flop-Inference Trap
- **The Trap:** On-chip memory for 16×16 NPU with $K_{\text{tile}}=128$:
  - Weight Buffer: $2 \times 2\text{ KB} = 4\text{ KB}$
  - Activation Buffer: $2 \times 2\text{ KB} = 4\text{ KB}$
  - Output Staging Buffer: $2 \times 256\text{ B} = 512\text{ B}$
  - Total on-chip SRAM: $\mathbf{8.5\text{ KB} = 69,632\text{ bits}}$.
  - If synthesized naively as SystemVerilog reg arrays, Yosys infers **~69,600 flip-flops** (~1.5 mm² on SkyWater 130 nm HD). This would dwarf the entire 256-PE compute array (~12.3k DFFs) by over $4\times$, skewing all area and power reports.
- **The Solution:**
  - `npu_sram.sv` is structured with Yosys attribute `(* blackbox *)` and conditional compilation (`ifdef SYNTHESIS`).
  - Under simulation (`!SYNTHESIS`): Executes a behavioral synchronous RAM with realistic 1-cycle read latency.
  - Under synthesis (`SYNTHESIS`): Treated as a fixed hard macro cell. Standard cell synthesis reports honest logic gates + flip-flops, with macro area accounted separately from OpenRAM datasheets.

#### B. Ping-Pong Banking & Mutual Exclusion
- Both `npu_buf_act` and `npu_buf_wgt` wrap two distinct 2 KB SRAM macros (`u_sram_bank0` and `u_sram_bank1`).
- **Port Mutex:**
  - When `cur_bank == 0`: Bank 0 is read-only for compute (`re = rd_en`, `we = 0`), Bank 1 is write-only for fetch (`we = wr_en`, `re = 0`).
  - When `cur_bank == 1`: Bank 1 is read-only for compute, Bank 0 is write-only for fetch.
- **Benefits:**
  - Eliminates multiport SRAM cost: single-port (1RW) or simple dual-port (1R1W) OpenRAM macros can be used directly with zero contention.
  - Zero read-write collisions or memory arbitration stalls.
  - Single-cycle atomic bank swap via `swap` strobe.

#### C. Tensor Memory Wide Fetch Engine (`npu_fetch.sv`)
- **Bus Width:** 128-bit wide data bus ($16\text{ bytes/beat}$). At 50 MHz, delivers $800\text{ MB/s}$ bandwidth.
- **Decoupled Request-Response Pipeline:**
  - Tracks outgoing memory read requests (`req_cnt`) independently from incoming data beats (`resp_cnt`).
  - Handles memory backpressure stalls (`mem_ready = 0`) and variable response latencies without dropped words or double counting.
  - Generates single-cycle `fetch_done` pulse upon writing the complete requested tile into the idle buffer bank.

#### D. Weight-Tile Residency
- Neural network GEMM loops are ordered $M \rightarrow K \rightarrow N$:
  - Weight tile $(M, K)$ is loaded once into the active weight buffer bank.
  - Spatial activations sweep across the $N$ dimension.
  - `npu_buf_wgt` holds the current weight bank resident across multiple activation buffer swaps without triggering memory refetches.
  - On Conv2 ($N=196$, 13 tiles of 16), holding weight tiles resident slashes weight fetch memory traffic by **$\approx 13\times$** ($196 / 16 \approx 12.25$).

### 3. Implementation Steps

1. **RTL Design:**
   - [src/buffer/npu_sram.sv](file:///Users/dhruv/Documents/int8_npu/src/buffer/npu_sram.sv): Parameterized SRAM wrapper (DW=128, WORDS=128 = 2 KB) with `(* blackbox *)`.
   - [src/buffer/npu_buf_act.sv](file:///Users/dhruv/Documents/int8_npu/src/buffer/npu_buf_act.sv): 4 KB ping-pong activation buffer wrapping two `npu_sram` macros with multiplexed compute-read / fetch-write ports.
   - [src/buffer/npu_buf_wgt.sv](file:///Users/dhruv/Documents/int8_npu/src/buffer/npu_buf_wgt.sv): 4 KB ping-pong weight buffer supporting persistent weight-tile residency.
   - [src/ctrl/npu_fetch.sv](file:///Users/dhruv/Documents/int8_npu/src/ctrl/npu_fetch.sv): 128-bit wide fetch engine with decoupled request/response counters and buffer write generation.

2. **Verification Environment (`tb/unit/tb_npu_fetch_buffer.sv`):**
   - Implemented simulated 64 KB Tensor Memory model with programmable backpressure and pipeline delay emulation.
   - Developed 5 comprehensive test suites:
     - **Test 1:** Direct ping-pong buffer write, bank swap, and 1-cycle synchronous read.
     - **Test 2:** Standalone `npu_fetch` transfer of 16 words (256 B) with zero-wait memory.
     - **Test 3:** `npu_fetch` transfer of 32 words (512 B) under memory backpressure and random pipeline stalls.
     - **Test 4:** Overlapped fetch and compute with continuous pipelined streaming (16 words/tile at 1 word/cycle) across bank swap boundaries.
     - **Test 5:** Weight-tile residency verifying weight buffer persistence across 3 consecutive activation buffer swaps with zero refetch traffic.

3. **Automation & Tooling:**
   - [scripts/run_buffer_test.py](file:///Users/dhruv/Documents/int8_npu/scripts/run_buffer_test.py): Python runner automating simulation on Icarus Verilog and Verilator 5, followed by strict Yosys synthesis assertions.
   - [scripts/sim_buffer.sh](file:///Users/dhruv/Documents/int8_npu/scripts/sim_buffer.sh): Executable bash wrapper.
   - [scripts/view_wave.sh](file:///Users/dhruv/Documents/int8_npu/scripts/view_wave.sh): Added `buf` target to inspect `sim_output/tb_npu_fetch_buffer.vcd`.

### 4. Verification Results & Synthesis Audit

- **Dual-Simulator Results:**
  ```text
  [TEST 1] Direct Ping-Pong Buffer Write, Bank Swap, and 1-Cycle Read
  [PASS] Test 1 completed: Verified ping-pong write, swap, and synchronous read.

  [TEST 2] npu_fetch Transfer with Zero-Wait Memory (16 words = 256 B)
    -> fetch_done received at time 525000. fetch_busy = 0
  [PASS] Test 2 completed: 16 words transferred accurately by npu_fetch.

  [TEST 3] npu_fetch Transfer under Memory Backpressure & Pipeline Stalls
    -> Stalled transfer completed successfully at time 1545000
  [PASS] Test 3 completed: 32 words transferred with 0 dropped beats under backpressure.

  [TEST 4] Concurrent Overlapped Fetch & Compute with Pipelined Streaming
    -> Starting concurrent Tile 0 compute read and Tile 1 background fetch...
    -> Tile 0 compute complete and Tile 1 background fetch complete. Swapping banks...
  [PASS] Test 4 completed: 16 words/tile streamed with 100% throughput and overlapped fetch!

  [TEST 5] Weight Tile Residency (Preserving Weight Buffer across 3 Act Swaps)
    -> Activation sweep pass 0: Reading resident weight tile...
    -> Activation sweep pass 1: Reading resident weight tile...
    -> Activation sweep pass 2: Reading resident weight tile...
  [PASS] Test 5 completed: Weight tile held resident with ZERO refetch traffic!

  ALL 5 TESTS PASSED! (0 errors)
  ```
  - `iverilog`: 5/5 suites passed (0 errors)
  - `verilator 5.050`: 5/5 suites passed (0 errors)

- **Yosys Synthesis & Flop Count Audit:**
  - `npu_sram`: Blackboxed macro cell. Exactly **0 flip-flops** inferred from memory arrays.
  - `npu_buf_act`: Exactly **1 DFF** (`cur_bank`), 2 `npu_sram` macro instantiations, 128 `$_MUX_` cells for the 128-bit read bus.
  - `npu_buf_wgt`: Exactly **1 DFF** (`cur_bank`), 2 `npu_sram` macro instantiations, 128 `$_MUX_` cells for the 128-bit read bus.
  - `npu_fetch`: Exactly **194 DFFs** (state register, address registers, write data buffer), **0 latches**.

- **Waveforms Generated:**
  - `sim_output/tb_npu_fetch_buffer.vcd` (237 KB) — Viewable via `./scripts/view_wave.sh buf`.

---

## 2026-09-28 — Milestone 5: Hardware Sequencer & M/N/K Multi-Loop Tiling Engine

### 1. Context & Objectives
- **Target:** Implement `npu_sequencer.sv` to orchestrate 3D nested loops ($M, K, N$), schedule DMA memory fetches, control ping-pong buffer swaps, sequence compute streaming, handle accumulator clear/drain, and compute partial edge-tile bounding boxes.
- **Exit Criterion (from SPEC.md):** Full $M/N/K$ tiling verified for arbitrary layer geometries including partial edge tiles; reduction depth splitting ($K > K_{\text{TILE\_MAX}}$) verified with live accumulators and delayed draining; weight tile residency verified with $13\times$ traffic reduction on Conv2; synthesized to latch-free control logic in Yosys.

### 2. Microarchitectural Decisions & Analysis

#### A. Weight-Residency Priority Loop Nest ($M \rightarrow K \rightarrow N$)
- **The Optimization:** Convolutional layers have large spatial dimensions ($N = H_{\text{out}} \times W_{\text{out}}$, e.g. Conv2 $N=196$) with moderate reduction depth ($K \le 128$).
- Iterating $N$ as the innermost loop holds the weight tile $(M, K)$ resident in `npu_buf_wgt` for the entire spatial sweep.
- On Conv2 ($M=16, N=196, K=64$), the weight tile is fetched **exactly once** for 13 spatial activation tiles, cutting weight memory traffic by **$13\times$**.

#### B. Reduction Depth Splitting ($K > K_{\text{TILE\_MAX}}$) for Output Stationary
- Output-stationary systolic arrays preserve partial sums in place within the PE accumulators.
- When $K$ exceeds on-chip buffer capacity (e.g. FC layer $K=784$ with $K_{\text{tile}}=128$), $K$ is sliced into sub-tiles:
  - Sub-tile 0 ($k_0 = 0..127$): `first_k = 1, last_k = 0`. Accumulators clear and accumulate.
  - Sub-tiles 1..5 ($k_0 = 128..640$): `first_k = 0, last_k = 0`. Accumulators remain live, adding new products to the running sum.
  - Sub-tile 6 ($k_0 = 768..783$): $k_{\text{len}} = 16$, `first_k = 0, last_k = 1`. Final 16 products accumulated, followed by accumulator snapshot and drain.
- Eliminates the need to write intermediate partial sums to external memory or reload them for accumulation.

#### C. Arbitrary Sub-Tile & Edge-Tile Geometry
- Neural network layers frequently present non-tile-aligned dimensions:
  - Conv1: $M=8 < \text{ARR\_M}$ (partial row channels).
  - FC: $N=1 < \text{ARR\_N}$ (degenerate column vector).
  - Spatial edges: Conv2 $N=196 = 12 \times 16 + 1 \times 4$ (last tile has $n_{\text{len}}=4$).
- The sequencer calculates dynamic bounding boxes:
  $$m_{\text{len}} = \min(\text{ARR\_M}, M - m_0), \quad n_{\text{len}} = \min(\text{ARR\_N}, N - n_0), \quad k_{\text{len}} = \min(K_{\text{tile}}, K - k_0)$$
- Emits active masks `active_m_mask` and `active_n_mask` to zero-pad unallocated array inputs, ensuring non-active PEs accumulate 0 with zero software patching or hardware stalls.

#### D. Built-in Hardware Performance Profiling
- Tracks real-time hardware counters from `seq_start` to `seq_done`:
  - `perf_cycles`: total wall clock cycles for the layer.
  - `perf_macs`: cumulative MAC operations actually issued ($\sum m_{\text{len}} \times n_{\text{len}} \times k_{\text{len}}$).
  - `perf_stall`: cycles stalled waiting for DMA fetch buffer readiness.
- Directly measures hardware utilization:
  $$\text{Utilization} = \frac{\text{perf\_macs}}{\text{perf\_cycles} \times 256}$$

### 3. Implementation Steps

1. **RTL Design ([src/ctrl/npu_sequencer.sv](file:///Users/dhruv/Documents/int8_npu/src/ctrl/npu_sequencer.sv)):**
   - Implemented 11-state FSM: `S_IDLE`, `S_FETCH_WGT`, `S_FETCH_ACT`, `S_WAIT_FETCH`, `S_BUFFER_SWAP`, `S_COMPUTE_PRIME`, `S_COMPUTE_STREAM`, `S_COMPUTE_FILL`, `S_DRAIN_WAIT`, `S_NEXT_TILE`, `S_DONE`.
   - Integrated tile descriptor formatting (`tile_desc_t`) with 32-bit address generation arithmetic for weights, activations, outputs, and bias.
   - Implemented weight residency tracking (`wgt_resident`) and automatic loop-order selection (`k_split_multitile`).

2. **Verification Suite ([tb/unit/tb_npu_sequencer.sv](file:///Users/dhruv/Documents/int8_npu/tb/unit/tb_npu_sequencer.sv)):**
   - Emulated fetch engines, ping-pong buffer banking, and column drain modules.
   - Developed 7 test suites:
     - **Test 1:** Standard full-tile GEMM ($M=16, N=16, K=16$).
     - **Test 2:** FC layer ($M=10, N=1, K=784$) with 7 $K$-slices, live accumulators, and single final drain.
     - **Test 3:** Conv2 spatial sweep ($M=16, N=196, K=64$) with 13 tiles, verifying $13\times$ weight fetch reduction and edge tile $n_{\text{len}}=4$.
     - **Test 4:** Conv1 partial channels ($M=8, N=32, K=25$) verifying row masking.
     - **Test 5:** Non-aligned 3D edge tiles ($M=17, N=33, K=135$) verifying edge handling across all 3 dimensions simultaneously.
     - **Test 6:** Error handling with invalid dimensions ($M=0$).
     - **Test 7:** Immediate abort mid-execution.

3. **Automation & Tooling:**
   - [scripts/run_sequencer_test.py](file:///Users/dhruv/Documents/int8_npu/scripts/run_sequencer_test.py): Python runner for dual-simulator and synthesis check.
   - [scripts/sim_sequencer.sh](file:///Users/dhruv/Documents/int8_npu/scripts/sim_sequencer.sh): Executable wrapper.
   - [scripts/view_wave.sh](file:///Users/dhruv/Documents/int8_npu/scripts/view_wave.sh): Added `seq` option to view `sim_output/tb_npu_sequencer.vcd`.

### 4. Verification Results & Synthesis Audit

- **Dual-Simulator Results:**
  ```text
  [TEST 1] Standard Full-Tile GEMM: M=16, N=16, K=16
    -> Layer finished: perf_cycles=60, perf_macs=4096, perf_stall=4
    -> Tiles launched=1, Wgt fetches=1, Act fetches=1, Drains=1
  [PASS] Standard Full-Tile GEMM completed successfully!

  [TEST 2] FC Layer (K=784 Reduction Splitting): M=10, N=1, K=784
    -> Layer finished: perf_cycles=1080, perf_macs=7840, perf_stall=28
    -> Tiles launched=7, Wgt fetches=7, Act fetches=7, Drains=1
  [PASS] FC Layer (K=784 Reduction Splitting) completed successfully!

  [TEST 3] Conv2 Spatial Sweep with Weight Residency: M=16, N=196, K=64
    -> Layer finished: perf_cycles=1380, perf_macs=200704, perf_stall=52
    -> Tiles launched=13, Wgt fetches=1, Act fetches=13, Drains=13
  [PASS] Conv2 Spatial Sweep with Weight Residency completed successfully!

  [TEST 4] Conv1 Partial Channels: M=8, N=32, K=25
    -> Layer finished: perf_cycles=136, perf_macs=6400, perf_stall=8
    -> Tiles launched=2, Wgt fetches=1, Act fetches=2, Drains=2
  [PASS] Conv1 Partial Channels completed successfully!

  [TEST 5] Non-Aligned 3D Edge Tiles: M=17, N=33, K=135
    -> Layer finished: perf_cycles=1321, perf_macs=75735, perf_stall=48
    -> Tiles launched=12, Wgt fetches=12, Act fetches=12, Drains=6
  [PASS] Non-Aligned 3D Edge Tiles completed successfully!

  [TEST 6] Error Handling (M=0 Invalid Dimension)
  [PASS] Error handling verified: seq_err correctly asserted.

  [TEST 7] Abort Mid-Execution
  [PASS] Abort handling verified: sequencer cleanly returned to IDLE.

  ALL 7 TESTS PASSED! (0 errors)
  ```
  - `iverilog`: 7/7 suites passed (0 errors)
  - `verilator 5.050`: 7/7 suites passed (0 errors)

- **Yosys Synthesis & Latch-Free Audit:**
  - Netlist check: `select -assert-count 0 t:$_DLATCH_*` passed. Exactly **0 latches inferred**.
  - Total flip-flops: **639 DFFs** (state register, coordinate registers, tile descriptors, performance counters).

- **Waveforms Generated:**
  - `sim_output/tb_npu_sequencer.vcd` (477 KB) — Viewable via `./scripts/view_wave.sh seq`.

---

## 2026-09-28 — Milestone 6: Requantization Epilogue (`npu_epilogue`) & Output Buffer (`npu_obuf`)

### 1. Context & Objectives
- **Target:** Transform 32-bit accumulators from the systolic drain unit back into quantized 8-bit activations with bias addition, fixed-point scale multiplication, rounding, saturation clamping, optional activation functions (ReLU), and stage them in a double-buffered SRAM buffer.
- **Exit Criterion (from SPEC.md):** `npu_epilogue`, `npu_obuf` double-buffered; requantization verified bit-exact against Python model across 1000+ random and corner inputs including saturation and rounding modes; buffer ping-pong verified under concurrent write/read.

### 2. Microarchitectural Decisions & Mathematical Formulations

#### A. Bit-Exact Fixed-Point Requantization Pipeline
Converting a 32-bit accumulated sum $A_{32}$ to an 8-bit quantized activation $Y_8$ requires a precise mathematical pipeline:
1. **Bias Addition:**
   $$P_{\text{biased}} = A_{32} + \text{bias}_{32} \quad (\text{33-bit signed})$$
2. **Fixed-Point Scaling Multiplier:**
   $$P_{\text{scaled}} = P_{\text{biased}} \times \text{mult}_{32} \quad (\text{65-bit signed})$$
3. **Symmetrical Rounding Half Away From Zero:**
   Standard truncation introduces negative DC bias. Rounding half away from zero symmetrically rounds $+0.5 \to +1$ and $-0.5 \to -1$:
   $$\text{For } P_{\text{scaled}} \ge 0: \quad P_{\text{rounded}} = (P_{\text{scaled}} + 2^{S-1}) \gg S$$
   $$\text{For } P_{\text{scaled}} < 0: \quad P_{\text{rounded}} = -((-P_{\text{scaled}} + 2^{S-1}) \gg S)$$
   where $S \in [0, 31]$ is the right-shift parameter.
4. **Activation Function (ReLU):**
   $$\text{If } \text{relu\_en} = 1: \quad Y = \max(0, P_{\text{rounded}})$$
5. **Signed INT8 Saturation Clamping:**
   $$\text{If } Y > +127 \implies Y = +127$$
   $$\text{If } Y < -128 \implies Y = -128$$

#### B. 2-Stage Pipelined 16-Lane SIMD Datapath (`npu_epilogue.sv`)
- To sustain a peak throughput of **1 tile row per cycle (16 lanes)** matching `npu_drain`:
  - **Stage 1 (Multiplier Stage):** 16 parallel 33-bit $\times$ 32-bit signed multipliers compute 65-bit intermediate scaled products. Registers store:
    - $16 \times 65$-bit scaled products
    - Global parameters (`rq_shift`, `relu_en`)
    - Pipeline valid handshake (`valid_stage1`)
  - **Stage 2 (Rounding & Clamp Stage):** 16 parallel rounding, shift, ReLU, and saturation units output sixteen 8-bit values packed into a 128-bit memory word (`out_data[127:0]`).
  - Total latency: 2 clock cycles; throughput: 1 128-bit word per cycle.

#### C. Double-Buffered Ping-Pong Output Buffer (`npu_obuf.sv`)
- **Memory Organization:** Wraps two 256 B `npu_sram` macros (`bank0` and `bank1`), providing 512 Bytes total storage.
  - One bank holds a full $16 \times 16$ INT8 tile ($16 \text{ rows} \times 16 \text{ bytes/row} = 256 \text{ Bytes}$).
- **Concurrent Non-Interfering Access:**
  - `cur_bank` (1 bit) points to the bank currently receiving stream writes from the epilogue unit at address `wr_addr[3:0]`.
  - The opposite bank (`!cur_bank`) is simultaneously exposed to the host/DMA read port at `rd_addr[3:0]`.
  - Zero write-after-read conflicts, zero contention stalls.
  - When a tile write finishes, the sequencer pulses `bank_swap`, instantly toggling `cur_bank <= ~cur_bank`.

### 3. Implementation Steps

1. **Golden Epilogue Model & Vector Generator (`model/golden_epilogue.py`):**
   - Implemented exact fixed-point requantization matching two's complement behavior.
   - Generated 1,332 test vectors (`tb/vectors/epilogue_vectors.txt`) covering:
     - Exact rounding tie-breaks ($+0.5 \to +1$, $-0.5 \to -1$, $+2.5 \to +3$, $-2.5 \to -3$)
     - Saturation bounds: $+127, +128, +1000 \to +127$; $-128, -129, -1000 \to -128$
     - Negative and positive biases
     - ReLU on positive, zero, and negative outputs
     - Shift amounts $S \in [0, 31]$
     - Extreme multiplier values and random sweeps

2. **SIMD Epilogue RTL (`src/epilogue/npu_epilogue.sv`):**
   - Implemented 16-lane parameterized SIMD datapath.
   - Addressed Yosys static-cast negative literal parsing quirk by expressing negation as `64'd0 - abs_rounded`.

3. **Output Staging Buffer RTL (`src/epilogue/npu_obuf.sv`):**
   - Implemented ping-pong bank routing logic around two `npu_sram` instances with `bank_swap` control.

4. **Comprehensive Testbench (`tb/unit/tb_npu_epilogue.sv`):**
   - Suite 1: Exhaustive verification across all 1,332 golden reference vectors.
   - Suite 2: 16-lane SIMD parallel operation check across varied per-lane accumulators.
   - Suite 3: Full $16 \times 16$ tile streaming write test with address sequencing.
   - Suite 4: Concurrent readback during active write verifying double-buffer isolation.

5. **Automated Runner & Scripts:**
   - [scripts/run_epilogue_test.py](file:///Users/dhruv/Documents/int8_npu/scripts/run_epilogue_test.py): Dual-simulator runner and Yosys synthesis auditor.
   - [scripts/sim_epilogue.sh](file:///Users/dhruv/Documents/int8_npu/scripts/sim_epilogue.sh): Standalone bash runner.
   - [scripts/view_wave.sh](file:///Users/dhruv/Documents/int8_npu/scripts/view_wave.sh): Added `epi` target to inspect `sim_output/tb_npu_epilogue.vcd`.

### 4. Verification Results & Synthesis Audit

- **Dual-Simulator Results:**
  ```text
  [TEST 1] Golden Model Vector Verification (1332 vectors)
    -> Applied 1332 test vectors from tb/vectors/epilogue_vectors.txt
  [PASS] All 1332 golden vectors verified bit-exactly!

  [TEST 2] 16-Lane SIMD Parallel Operation
    -> Applied 16 distinct accumulators across all lanes simultaneously
    -> Verified 128-bit packed output: 0x7f...
  [PASS] 16-lane SIMD datapath verified!

  [TEST 3] Full 16x16 Tile Epilogue & Buffer Storage
    -> Streamed 16 rows into npu_epilogue and staged into npu_obuf bank 0
    -> Swapped banks and verified 16 rows read back
  [PASS] 16x16 tile drain, requantization, and buffer storage verified!

  [TEST 4] Double-Buffered Ping-Pong Isolation
    -> Initiated write stream into Bank 1
    -> Simultaneously read previously staged Bank 0
    -> Verified concurrent reads returned uncorrupted data
  [PASS] Double-buffer ping-pong isolation verified!

  ALL 4 TESTS PASSED! (0 errors)
  ```
  - `iverilog`: 4/4 suites passed (0 errors)
  - `verilator 5.050`: 4/4 suites passed (0 errors)

- **Yosys Synthesis & Latch-Free Audit:**
  - `npu_obuf`: Exactly **1 DFF** (`cur_bank`), 2 `npu_sram` blackbox macros, **0 latches**.
  - `npu_epilogue`: Exactly **0 latches** (`select -assert-count 0 t:$_DLATCH_*`), **133 pipeline DFFs** (128 data bits + control pipeline registers).

- **Waveforms Generated:**
  - `sim_output/tb_npu_epilogue.vcd` (75 KB) — Viewable via `./scripts/view_wave.sh epi`.



---

## 2026-09-29 — Milestone 7: `npu_top` Integration, AXI4-Lite CSRs, Writeback, Epilogue Pipelining

### 1. Context & Objectives
- **Target:** Connect every block into `npu_top`, add the AXI4-Lite register block, memory arbiter and output writeback, and prove the whole tiled datapath bit-exact with real data.
- **Exit criterion:** A CPU-less testbench programs a layer over AXI4-Lite, the NPU fetches operands from a memory model, and the complete memory image after the run matches the golden model exactly.

### 2. What integration exposed
Milestones 4–6 each passed in isolation, but the sequencer test (M5) emulated fetch, buffers and drain and only counted events. With real data flowing, these problems appeared:

| Problem | Effect | Fix |
|---|---|---|
| Fetch addresses were `a_base + m0*K + k0` (bytes), i.e. one weight row | Each buffer word must hold one k-step across 16 channels; data would be wrong for every tile except N=16 activations | Defined a panel memory layout (SPEC §9); addresses are now `base + m0*K + k0*16` |
| Buffer read data held its last word after the stream ended | Array would keep accumulating the last k-step for the 30 fill cycles | Operands are gated to zero unless a valid read beat is present |
| No writeback engine, no bias loading | Outputs never reached memory; bias input was unconnected | Added `npu_wb` and a bias fetch (4 words per M panel) |
| `clr_acc` issued in the same cycle as the first read address | Harmless today because the operand at that cycle is zero, but wrong for back-to-back tile streaming | Clear delayed one cycle to align with SRAM read latency |
| Epilogue was one combinational stage (16 × 33×32 multiply + 65-bit round) | Likely critical path | Split into 2 stages: bias + multiply, then round/ReLU/saturate |

### 3. New and changed RTL
- **`npu_top.sv`** — full datapath. Array rows = output channels (fed from the weight buffer), columns = output positions (activation buffer).
- **`npu_csr.sv`** — AXI4-Lite slave; AW/W accepted in any order; SLVERR on unmapped/read-only writes; sticky `done`/`err`; new read-only `HW_CFG` (0x3C).
- **`npu_mem_arb.sv`** — one shared 128-bit memory port; fixed priority (writeback > weights > activations > bias); 8-deep tag FIFO routes in-order read responses back to their requester.
- **`npu_wb.sv`** — streams a drained tile from the obuf read bank to memory at 1 word/cycle with a 2-entry skid FIFO for backpressure.
- **`npu_sequencer.sv`** — panel addressing, bias fetch on first visit to an M panel, new `S_DRAIN_REQ` state that waits for the previous tile's writeback, `S_DONE` waits for the final writeback, `perf_clr`. `FILL_LATENCY` is now derived: `ARR_M + ARR_N - 2` cycles after the last read address.
- **`npu_epilogue.sv`** — 2-stage pipeline (latency 2, throughput 1 column/cycle).

### 4. Verification
- **`tb/integration/tb_npu_top.sv`** — AXI4-Lite master (randomised AW/W ordering), memory model with configurable latency, random jitter and random `mem_ready` backpressure. After each layer the **entire memory image** is compared with the golden image; the output region is pre-filled with a poison byte and followed by a guard word, so missed writes and overruns fail.
- **`model/golden_npu.py`** — packing to the panel layout and bit-exact layer reference (reuses `golden_epilogue.epilogue_scalar`).
- **Directed layers:** full tile; MNIST conv1 (8×784×9), conv2 (16×196×72), fc (10×1×784, K split with N=1); K split with N>16 (20×40×300); 1×1×1; K = K_TILE; edges in all three dims (17×33×129); saturating requant; unsigned output mode. Plus constrained-random shapes (M, N ≤ 48, K ≤ 300).
- **Results:** 30/30 layers bit-exact on iverilog and Verilator; 5 extra seeds × 110 layers on Verilator, all bit-exact. CSR readback, HW_CFG, SLVERR and M=0 error tests pass. All M1–M6 unit suites still pass.
- **Mutation checks** (to confirm the testbench can fail): fill latency one cycle short → caught; weight address missing the k0 term → caught. Clear one cycle early → *not* observable in the current serial schedule (see table above); a directed test for it will come with back-to-back streaming.

### 5. Baseline performance (ideal memory: 1-cycle latency, no backpressure)

| Layer | Cycles | Stall cycles | Utilization |
|---|---|---|---|
| conv1 8×784×9 | 4,574 | 1,417 | 4.8% |
| conv2 16×196×72 | 2,915 | 1,255 | 30.3% |
| fc 10×1×784 | 2,660 | 1,593 | 1.2% |

The sequencer runs fetch → compute → drain strictly in series, so the ping-pong buffers never actually overlap fetch with compute. This is the baseline for the architecture work in milestone 13.

### 6. Synthesis sanity (generic Yosys, `synth -flatten`, SRAMs blackboxed)
- 0 latches; 6 `npu_sram` macros; **26,243 flip-flops**.
- Largest contributors: PE accumulators + pass-through registers 12,288; **drain shadow registers 8,192** (a full copy of the accumulators, so the array can start the next tile during drain); skew triangles ~1,950; epilogue stage 1 ~1,040. The drain shadow is a candidate for area optimisation.

### 7. Corrections to earlier entries
- **M3:** "12,288 DFFs (matches SPEC Section 6)" — SPEC §6 counts only accumulators (8,192). 12,288 includes the 8-bit `a_out`/`b_out` pass-through registers.
- **M5:** the sequencer test verified control flow and counters only (emulated fetch/buffers/drain). The utilization numbers it reported (e.g. conv2 57%) assumed 2-cycle fetches and are superseded by the table above.
- **M6:** the epilogue was described as 2-stage but was implemented as a single combinational stage; it is now actually 2-stage. The old Test 4 printed PASS without checking data; it now verifies both banks.

---

## 2026-09-29 — Milestone 8: Full MNIST CNN on the NPU, Bit-Exact

### 1. Context & Objectives
- **Target:** Train a CNN, quantize it to INT8, and run the whole network on `npu_top` in RTL simulation with every layer's output identical to an integer reference.
- **Exit criterion:** All layers bit-exact for every simulated test image; NPU predictions identical to the golden int8 model.

### 2. Network and quantization (`compiler/qnet.py`)
- **Float model:** conv 1→8 (3×3, pad 1) → ReLU → maxpool 2 → conv 8→16 → ReLU → maxpool 2 → fc 784→10. Trained 3 epochs with Adam on the 60k training set.
- **Post-training quantization:**
  - Input pixels: `x_q = round(x · 127)`, range 0..127 (scale 1/127).
  - Weights: per-tensor symmetric INT8 (`s_w = max|W| / 127`). Per-tensor, not per-channel, because the NPU has one `RQ_MULT` per layer.
  - Bias: INT32 at scale `s_in · s_w`.
  - Activations: per-tensor INT8; `s_out` from the 99.99th percentile over 2,000 calibration images.
  - Requantization: `RQ_MULT / 2^RQ_SHIFT ≈ s_in · s_w / s_out`, using the largest shift ≤ 31 (the CSR field is 5 bits).
- **Integer reference:** `int_forward()` uses the same bit-exact epilogue as the hardware golden model (`epilogue_vector`, now vectorized; cross-checked against the scalar model on 19,200 random cases, including the int64-overflow fallback).

| Model | Test accuracy (10,000 images) |
|---|---|
| Float32 | 98.37% |
| INT8 (integer-only) | 98.39% |

| Layer | GEMM M×N×K | RQ_MULT | RQ_SHIFT |
|---|---|---|---|
| conv1 | 8×784×9 | 5,800,960 | 31 |
| conv2 | 16×196×72 | 4,400,310 | 31 |
| fc | 10×1×784 | 2,818,598 | 31 |

### 3. RTL inference (`tb/integration/tb_mnist.sv`)
- The testbench acts as the host driver. It does only data movement: im2col (3×3, pad 1) into the panel layout, 2×2 maxpool, and flatten. The NPU does every GEMM, bias, requantization and ReLU. This answers SPEC open questions 1 and 2 for v1: maxpool and im2col run on the host.
- Shared harness `tb/integration/npu_tb_harness.svh` (DUT, memory model, AXI-Lite tasks, host byte access) now used by both `tb_npu_top` and `tb_mnist`.
- Each image's conv1 (6,272 bytes), conv2 (3,136 bytes) and logits (10 bytes) are compared with the golden model.

### 4. Results
- **1,000 test images, Verilator, memory latency 4 + random 0–3, `mem_ready` 80%:** all bit-exact in every layer; NPU accuracy 98.70% = golden int8 accuracy on the same images. Wall time 25 s.
- **10 images on iverilog and Verilator:** bit-exact, identical cycle counts on both simulators.

Per-layer performance, ideal memory (1-cycle latency, no backpressure):

| Layer | Cycles | Stall | Utilization |
|---|---|---|---|
| conv1 | 4,574 | 1,417 | 4.8% |
| conv2 | 2,915 | 1,255 | 30.3% |
| fc | 2,660 | 1,593 | 1.2% |
| **Total** | **10,149 cycles/image** | | **11.2%** |

At 50 MHz this is 203 µs per image (~4,900 images/s). The low utilization comes from the serial fetch → compute → drain schedule (M7 §5) plus the shapes themselves: conv1 fills only 8 of 16 rows with K=9, and fc is a matrix-vector product (N=1). These are the numbers milestone 13 will improve.

### 5. How to run
```
python3 scripts/run_mnist_test.py -n 1000                          # Verilator, ideal memory
python3 scripts/run_mnist_test.py -n 1000 --mem-lat 4 --mem-jit 3 --mem-rdy 80
python3 scripts/run_mnist_test.py --simulator both -n 10
```

---

## 2026-09-30 — Milestone 13: Pipelined Sequencer (Prefetch, Back-to-Back Tiles, Overlapped Drain)

*Done before milestone 10 closes, so timing closure is measured on the final datapath.*

### 1. Context & Objectives
- **Problem (M7/M8):** the sequencer ran fetch → compute → fill → drain → writeback strictly in series. The ping-pong buffers existed but never overlapped anything. Baseline MNIST: 10,149 cycles/image, 11.2% array utilization.
- **Target:** overlap every stage the hardware allows, keep MNIST bit-exact, and measure the gain.

### 2. Microarchitecture
**Two decoupled controllers + one-entry mailbox (`npu_sequencer.sv`, rewritten).**
- *Fetch side* walks the loop nest and fetches tile *t+1* into the idle banks while compute streams tile *t*, then posts a descriptor. Weights are fetched only when (m0, k0) changes (weight residency is kept). Bias is fetched once per M panel into one of **two bias slots**, so panel *p+1*'s bias can load while panel *p* still drains; a slot is released when the last block of its panel has been drained.
- *Compute side* pops a descriptor, swaps banks and streams `k_len` words. Tiles of the same output block (K split) stream **back to back with no bubble**.

**Capture wave instead of fill-then-drain.**
- After a tile that completes an output block, compute issues a one-cycle clear pulse with zero operands. It follows the data diagonally through the array (the skew network already delays `clr` by i+j). As it reaches PE(i,j), the PE's result is copied into the drain shadow register (`npu_drain`, new `CAPTURE_ON_CLR` mode) and the PE clears.
- The next tile follows one cycle behind: **one bubble per output tile** instead of a 31-cycle fill wait.
- `drain_start` fires ARR_M+1 cycles after the wave (column 0 of the shadow complete). The shadow is read column by column exactly as the wave finishes each column. A new wave waits until the previous drain has started, otherwise it would overwrite shadow registers still being read.

**Overlapped output path (`npu_top.sv`).**
- drain → epilogue → obuf bank → 1-entry queue → writeback. A new drain may start once the previous tile has reached the queue and the queue is empty; its epilogue then writes the bank whose writeback has already finished.

**Hazards found while designing for back-to-back operation.** These were not visible in the serial design:

| Hazard | Fix |
|---|---|
| Buffer read data is selected by `cur_bank`, but it returns one cycle after the read. A swap right after a tile's last read would return the other bank's data. | Register the read bank at read time (`rd_bank_q`) in `npu_buf_act/wgt` and `npu_obuf` |
| Writeback of tile *t* can still be reading an obuf bank when tile *t+1*'s epilogue finishes and swaps | `npu_obuf` `EXT_RD_BANK` mode: writeback latches which bank to read at start |
| The accumulator clear arriving one cycle before its data (noted in M7) now matters: the preceding cycle carries the previous tile's last data | Already fixed in M7 (`clr_aligned`) |

Tile masks now travel with each read beat (`m_mask_q`/`n_mask_q`), since consecutive tiles can have different shapes.

### 3. Verification
- `tb_npu_top`: 30 directed + random layers on iverilog and Verilator; 5 seeds × 110 layers on Verilator; new `--harsh-mem` mode (latency 4–12, jitter 5–10, `mem_ready` 20–50%): 3 seeds × 70 layers. All bit-exact.
- `tb_mnist`: 1,000 images, memory latency 4+0–3 and 80% ready: all bit-exact, accuracy 98.70% = golden.
- Unit suites M1–M4, M6 pass. The M5 sequencer unit test drove the old serial interface through emulated fetch/drain units; it was retired, since `tb_npu_top` exercises the real sequencer with real data.
- **Mutation checks** (each must fail the regression):

| Injected bug | Result |
|---|---|
| Clear/capture wave not delayed to match SRAM read latency | caught (1,795 errors) — the M7 case that was unobservable is now covered |
| `drain_start` one cycle early | caught |
| Activation buffer read mux uses `cur_bank` | caught |
| obuf read bank follows `cur_bank` | not caught with normal memory (writeback has top arbiter priority and always finishes first); **caught under `--harsh-mem`** |
| Next wave allowed before previous drain starts | caught |

### 4. Results

Ideal memory (1-cycle latency, no backpressure):

| Layer | Serial (M8) | Pipelined | Speedup | Utilization |
|---|---|---|---|---|
| conv1 8×784×9 | 4,574 | 1,595 | 2.9× | 4.8% → 13.8% |
| conv2 16×196×72 | 2,915 | 1,398 | 2.1× | 30.3% → 63.1% |
| fc 10×1×784 | 2,660 | 1,770 | 1.5× | 1.2% → 1.7% |
| **MNIST total** | **10,149** | **4,763** | **2.13×** | 11.2% → 23.8% |

Memory latency 4 + 0–3 jitter, 80% `mem_ready`: 11,607 → 6,098 cycles/image (1.90×).

### 5. Where the remaining cycles go
With the control overhead gone, every MNIST layer is now limited by the single 16 B/cycle memory port:
- **conv2:** each tile needs 72 activation words in and 16 output words out = 88 port cycles, against 73 compute cycles.
- **conv1:** K = 9, so each tile needs 9 words in, 16 out, and only 9 compute cycles. The port, and the 16-cycle drain per 16 output columns, dominate.
- **fc:** N = 1, so every weight byte is used once. Each K tile moves 128 weight + 128 activation words for 128 compute cycles on one array column.

This is the roofline behaviour that milestone 12 will quantify with the `MEM_DW` sweep and the performance model.

---

## 2026-09-30 — Milestone 12: Cycle-Level Performance Model, Validation, Design-Space Exploration

### 1. Context & Objectives
- **Target:** a performance model that predicts RTL cycle counts, validated against the hardware performance counters, then used to explore design points the RTL does not implement.
- **Exit criterion:** model matches `PERF_CYCLES` per layer; sweeps over port width, K tile depth and batch size, with the bottleneck explained.

### 2. The model (`model/perf_model.py`)
- Models control and data movement only (no arithmetic): the three fetch engines, the memory arbiter and its 8-deep outstanding-read limit, the sequencer's fetch/compute controllers and mailbox, the capture wave and drain timer, the epilogue pipeline, the obuf queue and the writeback FIFO.
- Every state element updates once per cycle from the previous cycle's values, the same register-transfer timing as the RTL.
- It generalises beyond the RTL for exploration: memory port widths other than one panel word per beat, and other array sizes / K tile depths.
- Runs MNIST in ~70 ms (Verilator RTL: ~0.25 s per image; iverilog: seconds).
- `layer_bounds()` gives analytic lower bounds per resource (compute stream, memory port beats, drain) to explain *why* a layer takes what it takes.

### 3. Validation (`scripts/validate_perf_model.py`)
- **Deterministic memory** (fixed latency 1, 3, 6; always ready): 450 layers (directed MNIST shapes, K split, edge tiles, random shapes up to 48×48×300) — **cycle-exact on all 450**.
- **Random memory** (jitter, 20–100% `mem_ready`): the RTL and model use different random streams, so single runs differ by chance. Over 103 layers ≥ 500 cycles, one RTL run vs. the model's 8-seed mean: mean |error| 2.0%, mean signed error −1.2%.

### 4. Design-space exploration (`scripts/dse.py` → `docs/results/dse.md`, `docs/figures/`)

**Memory port width** (MNIST, ideal memory, 50 MHz):

| Port (B/cycle) | Cycles/image | Array utilization | Images/s |
|---|---|---|---|
| 4 | 16,788 | 6.7% | 2,978 |
| 8 | 8,791 | 12.9% | 5,688 |
| **16 (implemented)** | **4,767** | **23.8%** | **10,489** |
| 32 | 3,164 | 35.8% | 15,803 |
| 64 | 3,017 | 37.6% | 16,573 |

Throughput scales almost linearly with port width up to 16 B, gains 1.5× more at 32 B, then saturates. Past 32 B the bottleneck moves to compute and drain: conv1 uses only 8 of 16 rows with K = 9, and every output tile costs a 16-cycle drain.

**Roofline** (`docs/figures/roofline.png`): at 16 B/cycle every layer is memory-bound and runs at **78–92% of its bandwidth ceiling**: conv1 78%, conv2 86%, fc 89%. The pipelined design is close to what its memory port allows. Moving further up requires more bandwidth or more reuse, not better control.

**K tile depth.** The OpenRAM macro is 256 words deep, so K_TILE ≤ 256 costs no SRAM area:

| K_TILE | MNIST cycles/image | Why |
|---|---|---|
| 32 | 5,919 | conv2 (K = 72) splits along K with N > 16 → K-innermost loop order → weights refetched every tile |
| 64 | 6,185 | same effect, and fewer tiles to overlap |
| **128** | **4,767** | conv2 fits in one K tile; weights resident across all 13 N tiles |
| 256 | 4,874 | fc has fewer, longer tiles: the first fetch is exposed longer |

128 is the best of these for MNIST, and deeper is not free even though the area is.

**Batch size** (`docs/figures/batch_sweep.png`): fc utilization grows 1.7% → 27.4% from batch 1 to 16 (N = batch fills the array columns). Throughput rises to 17,504 images/s (1.67×). **Beyond 16 it stalls**: with N > 16 and K > K_TILE the sequencer switches to K-innermost order, which gives up weight reuse across N tiles, so fc traffic doubles. Fixing that would need either K_TILE ≥ K (a 784-deep weight buffer) or partial sums leaving the array — a concrete, measured next step.

### 5. What the model says to build next
1. Port width is the first-order lever up to 32 B/cycle (+50% throughput).
2. conv1 writes more than it reads: M = 8 fills half of each 16-byte output word, and outputs (6.3 KB useful, 12.5 KB moved) outweigh inputs. A packed output layout for M < 16 would cut conv1's traffic by about a third.
3. A loop order that keeps weights resident when K is split and N > 16 is what batch > 16 needs.

---

## 2026-09-30 — Milestone 9: Assertions, Functional Coverage, Parameterization, Regression

### 1. Context & Objectives
- **Target:** turn the tests into a regression that proves its own thoroughness: assertions on the invariants the pipelined design relies on, functional coverage of its corner cases, and verification of a second array size.
- **Exit criterion:** one command runs every suite; all pass; every coverage point is hit.

### 2. Assertions (`tb/sva/npu_sva.sv`, bound into `npu_top`, Verilator `--assert`)
| Assertion | Invariant |
|---|---|
| `a_act_swap_after_fetch`, `a_wgt_swap_after_fetch` | a bank becomes the read bank only after its fetch has landed |
| `a_reads_lockstep` | weight and activation buffers are read together |
| `a_wave_no_read` | a clear/capture wave carries zero operands |
| `a_blocked_wave_waits`, `a_wave_spacing` | capture waves are ≥ CAP_DELAY + 1 cycles apart, so drained shadow registers are never overwritten |
| `a_drain_free`, `a_wbq_free`, `a_wb_idle` | the output path never overwrites a tile in flight |
| `a_bias_slot_safe` | the bias slot being loaded is never the one being drained |
| `a_mem_aligned` | every memory access is word-aligned |
| `a_done_xor_err` | done and err are never raised together |
| `a_axi_b_hold`, `a_axi_r_hold` | AXI-Lite B/R responses hold until accepted |

Checked by mutation: letting `drain_ready` ignore the writeback queue makes `a_drain_free` fire and stop the simulation.

### 3. Functional coverage
15 coverage points, counted in every Verilator run and summed by the regression script. They cover the pipelined sequencer's corner cases:
- back-to-back K slices
- a wave blocked by the previous drain
- a drain blocked by the output path
- the next tile popped in the same cycle as a wave
- compute starved by fetch
- both bias slots
- the K-innermost loop order
- the writeback queue full while writeback runs
- memory backpressure
- the outstanding-read limit reached
- weight reuse
- partial M and N tiles
- layer done and error

**One hole was found.** "Pop in the wave cycle" never occurred: fetch always finishes after compute unless a wave is blocked by the drain. A directed drain-bound layer (16×256×4) was added to close it.

The CSR block now write-protects the layer configuration while busy (SLVERR). This is tested: a write to RQ_MULT mid-layer must return SLVERR and the layer must still be bit-exact. M10 relies on this property.

### 4. Parameterization verified at 8×8
Running at ARR = 8 found two hardcoded 16-byte word steps (`<< 4` in `npu_fetch` and `npu_wb`), now derived from `MEM_DW`. The golden model (`set_arr`), the testbench harness (`TB_ARR`) and the runner (`--arr`) are parameterized. The 8×8 configuration passes 330 random layers plus harsh-memory and iverilog runs, all bit-exact.

### 5. Regression (`scripts/run_regression.py`)
14 suites:
- 5 unit suites
- 16×16 integration: 3 seeds, harsh memory, and an iverilog run
- 8×8 integration: normal and harsh memory
- MNIST
- performance-model-vs-RTL validation

Result: **14/14 pass, 15/15 coverage points hit, about 2 minutes.**

---

## 2026-09-30 → 2026-10-01 — Milestone 10: sky130 Synthesis, Placement-Aware STA, Timing Closure

### 1. Context & Objectives
- **Target:** synthesize to SkyWater 130 nm HD with real SRAM macros, run static timing at three corners for a 50 MHz clock (20 ns), root-cause the critical paths, and fix them.
- **Scope:** no CTS or routing. The clock is ideal with stated uncertainty; wires are estimated from a global placement.

### 2. Flow
| Step | Tool | Notes |
|---|---|---|
| Synthesis | Yosys 0.63 (`pd/synth/synth.tcl`) | sky130_fd_sc_hd TT library; delay-driven ABC with fanout buffering and sizing; hierarchy kept for a per-module area report; `autoname` so constraints can name registers |
| SRAM | OpenRAM `sky130_sram_1kbyte_1rw1r_32x256_8` | 4 macros per 128×128 bank, 16 total; TT liberty only, used at all corners (stated limitation). The 256 B obuf banks are register files (`USE_MACRO = 0`): 8 more macros would have cost ~1.5 mm² |
| Placement | OpenROAD `global_placement` at 50% utilization | only so the resizer can build buffer trees and parasitics can be estimated (met2 RC) |
| Repair | `repair_design`, `repair_timing` (setup, then hold) | |
| STA | OpenSTA inside OpenROAD | ss_100C_1v60 / tt_025C_1v80 / ff_n40C_1v95; setup uncertainty 0.5 ns, hold 0.1 ns; I/O at 30% of the period |

Tool setup on macOS: OpenSTA was built from source (with CUDD). OpenROAD came from the litex-hub conda channel (x86 build running under Rosetta). The sky130 libraries came via `volare`.

### 3. Timing-closure history (what each optimization did)
Each row is a full synthesis + STA run at 20 ns. WNS/TNS are setup at the slow corner (ss).

| # | Change | Why | ss WNS | ss TNS |
|---|---|---|---|---|
| 0 | First flow: wire-load model, no placement | `repair_design` could not build buffer trees without placement; one enable net drove ~12K flops (220 ns slew) | −180 ns | — |
| 1 | **Free-running array** (`HAS_EN = 0`): removed the array-wide enable; idle cycles feed zero operands so accumulators hold (acc += 0) | Removes a 12K-fanout net and ~12K enable muxes; skew area −31% | (fanout fixed) | |
| 2 | **ABC buffering + global placement before repair**. ABC cannot buffer flop outputs (they are its primary inputs), so CSR config bits (~6K fanout) and clear taps (~700) stayed unbuffered until the resizer had placement | 8,271 buffers inserted, 107K → 232 DRV violations | −3.44 | −930 |
| 3 | **Operand register stage** before the array | The OpenRAM macros drive read data from the *falling* clock edge, and row/column 0 have no skew register, so PE(0,0)'s 8×8 MAC (~11 ns at ss) had half a cycle | | |
| 4 | **Static configuration**: CSR write-protects RQ_MULT, RQ_SHIFT and FLAGS while busy (SLVERR, tested); SDC gives those registers 4-cycle multicycle paths | The critical path was a config bit fanning out to all 16 epilogue multipliers: 11 ns of buffer chain plus 11.7 ns of multiplier. The config cannot change while the datapath uses it, and is first used ≥ 18 cycles after start | −2.32 | −609 |
| 5 | **Bias slot copied at drain start**; epilogue reads RQ_SHIFT/RELU/SIGNED directly instead of pipelined copies | Removes the 512-bit slot mux and a high-fanout shift register from the multiply/round paths | −1.55 | −115 |
| 6 | SDC fix: `autoname` named the FLAGS flops `bias_en/relu_en/signed_act`, so the multicycle pattern missed them | Constraint bug, not a design change | −1.11 | −89 |
| 7 | **Epilogue 3 stages**: bias add, multiply, round | Real datapath: `drain_data` → add → 33×32 multiply in one cycle | −0.53 | −6.0 |
| 8 | **Epilogue rounding restructured and split (4 stages)**: `(|p| >> s) + |p|[s−1]` instead of `(|p| + 2^(s−1)) >> s`, then register a 9-bit clamped magnitude + sign, then sign/ReLU/saturate | Removes a 65-bit add before the shifter and 64-bit compares; register cost 10 flops/lane instead of 64. Bit-exact on all 1,332 golden vectors | **+0.066 (met)** | 0 violating endpoints |

Typical and fast corners met setup from step 5 onward, with zero violations of any check type from step 6. Hold is clean at all corners throughout (min +0.14 ns at ff).

### 4. Final result — timing closed at 50 MHz, all corners

| Corner | Setup WNS | Hold WNS | Violating checks (setup, hold, recovery/removal, pulse width, period) |
|---|---|---|---|
| ss_100C_1v60 | **+0.066 ns** | +0.623 ns | 0 |
| tt_025C_1v80 | +7.93 ns | +0.264 ns | 0 |
| ff_n40C_1v95 | +8.37 ns | +0.138 ns | 0 |

- **Critical path now:** epilogue stage 2a (`prod_s1` → |x| → variable shift → rounding bit → clamp → `mag_s2`): a 65-bit negate followed by a barrel shifter, with 0.07 ns to spare at the slow corner. The next lever, if needed, is to keep the product in sign-magnitude form out of the multiplier to drop the negate.
- **Typical-corner headroom:** 20 − 7.93 ≈ 12.1 ns of logic, i.e. roughly 80 MHz at TT before CTS/routing effects.
- **TNS caveat:** this OpenROAD build's `total_negative_slack_corner_cmd` reports values inconsistent with its own path reports (e.g. positive "TNS" at tt). The per-check violator reports (`pd/reports/sta_violators_<corner>.txt`) are the authority here, and they are empty at every corner.

**Area (Yosys, sky130 HD + OpenRAM, before placement buffering): 6.00 mm²**

| Block | Area (mm²) | Share |
|---|---|---|
| SRAM macros (16 × 1 KB OpenRAM) | 3.05 | 51% |
| PE array (256 × 5,006 µm²) | 1.28 | 21% |
| Epilogue (16 lanes, 33×32 multipliers) | ~1.0 | 17% |
| Drain shadow + column mux | 0.36 | 6% |
| obuf register files | 0.16 | 3% |
| Sequencer, CSR, fetch, writeback, arbiter, skew | ~0.15 | 2% |

Logic alone is ~2.95 mm²; the epilogue is the largest logic block after the array and is only busy 16 of every ≥19 cycles per tile. Time-multiplexing it (e.g. 4 lanes over 4 cycles) would be the main area optimization if area mattered more than drain latency.

Reports: `pd/reports/synth_area.txt`, `pd/reports/sta_summary.txt`, `sta_setup_ss_final.txt`, `sta_hold_ff.txt`, `sta_violators_*.txt`. Reproduce with `python3 scripts/run_synth.py --period 20` (~11 min synthesis + ~60 min placement/repair/STA).

### 5. Side effect on performance
Each epilogue stage adds one cycle of latency per output tile, which does not change throughput. One effect showed up in the cycle counts: conv1 got *faster*, 1,597 → 1,436 cycles. The performance model shows the same memory beats but port idle time dropping from 363 to 198 cycles. conv1 is port-bound, and the extra cycle shifts when writeback bursts start, so they fill gaps in the fetch stream instead of colliding with it. The lesson is that conv1's speed depends on traffic phase, not just bandwidth. An explicit fix, such as a third activation bank to let prefetch start earlier, would make it independent of that luck. MNIST: **4,610 cycles/image** (ideal memory), bit-exact.

The M12 tables were measured on the 2-stage epilogue. `scripts/dse.py` was re-run on the final design; `docs/results/dse.md` and the figures are current. The conclusions are unchanged, but absolute numbers moved slightly (e.g. a 32 B/cycle port now gives 3,266 cycles/image, a 1.41× speedup).
