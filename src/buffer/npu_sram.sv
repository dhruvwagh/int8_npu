// ============================================================================
// File: npu_sram.sv
// Description: Parameterized 1R1W SRAM Wrapper
//              - Simulation: behavioral memory, synchronous 1-cycle read.
//              - Synthesis, USE_MACRO = 1: DW/32 OpenRAM sky130 macros
//                (sky130_sram_1kbyte_1rw1r_32x256_8) side by side. Port 0 (RW)
//                is used for writes, port 1 (R) for reads. Keeps area honest:
//                no flip-flops are inferred for the storage array.
//              - Synthesis, USE_MACRO = 0: flip-flop register file. Used for
//                small buffers where the smallest macro (1 KB, 0.19 mm2) would
//                be mostly empty.
//              - FPGA (`define FPGA): the behavioral model, which FPGA synthesis
//                infers as block RAM.
//              Read data is only guaranteed in the cycle after the read
//              (the OpenRAM macro does not hold dout), which is all the NPU uses.
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_SRAM_SV
`define NPU_SRAM_SV

`timescale 1ns/1ps

module npu_sram #(
  parameter int DW        = 128,            // Data width (128 bits = 16 bytes)
  parameter int WORDS     = 128,            // Depth (128 words = 2 KB)
  parameter int AW        = $clog2(WORDS),  // Address width
  parameter bit USE_MACRO = 1'b1            // Synthesis: 1 = OpenRAM macros, 0 = flops
) (
  input  logic          clk,
  input  logic          csb,            // Active-low chip select
  // Write Port
  input  logic          we,             // Active-high write enable
  input  logic [AW-1:0] waddr,          // Write address
  input  logic [DW-1:0] wdata,          // Write data
  // Read Port
  input  logic          re,             // Active-high read enable
  input  logic [AW-1:0] raddr,          // Read address
  output logic [DW-1:0] rdata           // Read data (synchronous 1-cycle latency)
);

`ifdef FPGA
  logic [DW-1:0] mem [WORDS];
  always_ff @(posedge clk) begin
    if (!csb && we) mem[waddr] <= wdata;
    if (!csb && re) rdata <= mem[raddr];
  end
`elsif SYNTHESIS
  generate
    if (USE_MACRO) begin : g_macro
      localparam int MACRO_DW = 32;
      localparam int MACRO_AW = 8;           // 256 words
      initial begin
        if (DW % MACRO_DW != 0 || AW > MACRO_AW)
          $fatal(1, "npu_sram: %0dx%0d does not map onto 32x256 macros", WORDS, DW);
      end

      for (genvar s = 0; s < DW / MACRO_DW; s++) begin : g_slice
        wire [MACRO_DW-1:0] dout0_unused;
        sky130_sram_1kbyte_1rw1r_32x256_8 u_macro (
          .clk0   (clk),
          .csb0   (csb | ~we),
          .web0   (1'b0),
          .wmask0 (4'hF),
          .addr0  (MACRO_AW'(waddr)),
          .din0   (wdata[s*MACRO_DW +: MACRO_DW]),
          .dout0  (dout0_unused),
          .clk1   (clk),
          .csb1   (csb | ~re),
          .addr1  (MACRO_AW'(raddr)),
          .dout1  (rdata[s*MACRO_DW +: MACRO_DW])
        );
      end
    end else begin : g_flops
      logic [DW-1:0] mem [WORDS];
      always_ff @(posedge clk) begin
        if (!csb && we) mem[waddr] <= wdata;
        if (!csb && re) rdata <= mem[raddr];
      end
    end
  endgenerate
`else
  // Behavioral simulation model
  logic [DW-1:0] mem [WORDS];

  always_ff @(posedge clk) begin
    if (!csb) begin
      if (we) begin
        mem[waddr] <= wdata;
      end
      if (re) begin
        rdata <= mem[raddr];
      end
    end
  end
`endif

endmodule : npu_sram

`endif // NPU_SRAM_SV
