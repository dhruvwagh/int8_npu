// ============================================================================
// File: npu_pkg.sv
// Description: Global package for INT8 Systolic Array NPU
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_PKG_SV
`define NPU_PKG_SV

`timescale 1ns/1ps

package npu_pkg;
  /* verilator lint_off UNUSEDPARAM */

  // --------------------------------------------------------------------------
  // Default Architectural Parameters (Overridable at top-level)
  // --------------------------------------------------------------------------
  localparam int DEFAULT_ARR_M   = 16;   // Number of PE rows
  localparam int DEFAULT_ARR_N   = 16;   // Number of PE columns
  localparam int DEFAULT_ACC_W   = 32;   // Accumulator bit width
  localparam int DATA_W          = 8;    // INT8 operand width
  localparam int PROD_W          = 16;   // Signed INT8 x INT8 product width
  localparam int MEM_DW          = 128;  // Tensor memory port width (bits)
  localparam int ADDR_W          = 32;   // Address bus width

  localparam int M_TILE          = 16;   // Default M tile dimension
  localparam int N_TILE          = 16;   // Default N tile dimension
  localparam int K_TILE_MAX      = 128;  // Maximum K tile depth per sub-reduction

  // --------------------------------------------------------------------------
  // AXI4-Lite CSR Register Map (Byte Offsets)
  // --------------------------------------------------------------------------
  localparam logic [7:0] CSR_CTRL        = 8'h00; // W:  [0] start, [1] abort, [2] perf_clr
  localparam logic [7:0] CSR_STATUS      = 8'h04; // R:  [0] busy,  [1] done,  [2] err
  localparam logic [7:0] CSR_DIM_M       = 8'h08; // RW: Output channels M
  localparam logic [7:0] CSR_DIM_N       = 8'h0C; // RW: Output spatial points N
  localparam logic [7:0] CSR_DIM_K       = 8'h10; // RW: Reduction depth K
  localparam logic [7:0] CSR_A_BASE      = 8'h14; // RW: Weight matrix base address
  localparam logic [7:0] CSR_B_BASE      = 8'h18; // RW: Activation matrix base address
  localparam logic [7:0] CSR_C_BASE      = 8'h1C; // RW: Output matrix base address
  localparam logic [7:0] CSR_BIAS_BASE   = 8'h20; // RW: Per-channel bias base address
  localparam logic [7:0] CSR_RQ_MULT     = 8'h24; // RW: Requantize multiplier (INT32)
  localparam logic [7:0] CSR_RQ_SHIFT    = 8'h28; // RW: Requantize right shift
  localparam logic [7:0] CSR_FLAGS       = 8'h2C; // RW: [0] relu_en, [1] bias_en, [2] signed_act
  localparam logic [7:0] CSR_PERF_CYCLES = 8'h30; // R:  Total execution cycles
  localparam logic [7:0] CSR_PERF_MACS   = 8'h34; // R:  MAC-cycles issued
  localparam logic [7:0] CSR_PERF_STALL  = 8'h38; // R:  Buffer stall cycles

  // Control and Status bit offsets
  localparam int CTRL_START_BIT     = 0;
  localparam int CTRL_ABORT_BIT     = 1;
  localparam int CTRL_PERF_CLR_BIT  = 2;

  localparam int STATUS_BUSY_BIT    = 0;
  localparam int STATUS_DONE_BIT    = 1;
  localparam int STATUS_ERR_BIT     = 2;

  localparam int FLAG_RELU_EN_BIT   = 0;
  localparam int FLAG_BIAS_EN_BIT   = 1;
  localparam int FLAG_SIGNED_ACT_BIT= 2;

  // --------------------------------------------------------------------------
  // Structs & Typedefs
  // --------------------------------------------------------------------------

  // Operand data types
  typedef logic signed [DATA_W-1:0] act_t;     // Activation element (INT8)
  typedef logic signed [DATA_W-1:0] wgt_t;     // Weight element (INT8)
  typedef logic signed [PROD_W-1:0] prod_t;    // Multiplier output (INT16)

  // Epilogue configuration
  typedef struct packed {
    logic signed [31:0] rq_mult;
    logic [4:0]         rq_shift;
    logic               relu_en;
    logic               bias_en;
    logic               signed_act;
  } epilogue_cfg_t;

  // Single tile execution descriptor (issued by sequencer to compute pipeline)
  typedef struct packed {
    logic [15:0]        m0;         // M coordinate origin
    logic [15:0]        n0;         // N coordinate origin
    logic [15:0]        k0;         // K coordinate origin
    logic [7:0]         m_len;      // Valid rows in this tile (<= ARR_M)
    logic [7:0]         n_len;      // Valid cols in this tile (<= ARR_N)
    logic [7:0]         k_len;      // Reduction length for this tile (<= K_TILE_MAX)
    logic               first_k;    // 1: clear accumulators at tile start
    logic               last_k;     // 1: drain accumulators & run epilogue
    logic [ADDR_W-1:0]  a_base;     // Weight buffer base
    logic [ADDR_W-1:0]  b_base;     // Activation buffer base
    logic [ADDR_W-1:0]  c_base;     // Output destination base
    logic [ADDR_W-1:0]  bias_base;  // Bias vector base
  } tile_desc_t;

  /* verilator lint_on UNUSEDPARAM */
endpackage : npu_pkg

`endif // NPU_PKG_SV
