# Design-space exploration (performance model)

Model: `model/perf_model.py`, cycle-exact against RTL at the implemented point (16x16 array, 16 B/cycle port, K_TILE = 128; 450 layers compared). Ideal memory (1-cycle latency, always ready). Clock 50 MHz for time figures.

## 1. Tensor-memory port width

| Port (B/cycle) | conv1 cycles | conv2 cycles | fc cycles | Total cycles/image | Array utilization | Images/s @50 MHz |
|---|---|---|---|---|---|---|
| 4 | 5,350 | 5,041 | 6,403 | 16,794 | 6.7% | 2,977 |
| 8 | 2,874 | 2,625 | 3,314 | 8,813 | 12.9% | 5,673 |
| 16 (RTL) | 1,436 | 1,401 | 1,773 | 4,610 | 24.6% | 10,846 |
| 32 | 1,186 | 1,077 | 1,003 | 3,266 | 34.7% | 15,309 |
| 64 | 1,177 | 1,039 | 903 | 3,119 | 36.3% | 16,031 |

## 2. Roofline (16 B/cycle port, 256 MAC/cycle peak)

Arithmetic intensity uses the traffic the NPU actually moves (operands, bias, outputs).

| Layer | Batch | MACs / byte | Achieved MAC/cycle | Bandwidth ceiling MAC/cycle | Fraction of ceiling |
|---|---|---|---|---|---|
| conv1 | 1 | 2.85 | 39.3 | 45.6 | 86% |
| conv2 | 1 | 11.68 | 161.2 | 186.9 | 86% |
| fc | 1 | 0.31 | 4.4 | 5.0 | 89% |
| conv1 | 16 | 2.88 | 41.5 | 46.0 | 90% |
| conv2 | 16 | 13.03 | 192.2 | 208.5 | 92% |
| fc | 16 | 4.94 | 70.2 | 79.0 | 89% |

## 3. K tile depth (buffer depth per bank)

The OpenRAM macro is 256 words deep, so K_TILE up to 256 uses the same 16 macros: the only cost is an 8-bit -> 9-bit read address.

| K_TILE | fc cycles | fc K tiles | Total cycles/image | SRAM macros |
|---|---|---|---|---|
| 32 | 1,810 | 25 | 5,762 | 16 |
| 64 | 1,751 | 13 | 6,028 | 16 |
| 128 (RTL) | 1,773 | 7 | 4,610 | 16 |
| 256 | 1,880 | 4 | 4,717 | 16 |

## 4. Batch size (N grows with batch; weights are reused across images)

| Batch | conv1 util | conv2 util | fc util | Cycles/image | Images/s @50 MHz |
|---|---|---|---|---|---|
| 1 | 15.4% | 63.0% | 1.7% | 4,610 | 10,846 |
| 2 | 15.8% | 69.3% | 3.5% | 3,554 | 14,067 |
| 4 | 16.0% | 73.0% | 6.9% | 3,028 | 16,513 |
| 8 | 16.1% | 74.4% | 13.8% | 2,774 | 18,025 |
| 16 | 16.2% | 75.1% | 27.4% | 2,647 | 18,887 |
| 32 | 16.2% | 75.4% | 27.9% | 2,637 | 18,960 |

