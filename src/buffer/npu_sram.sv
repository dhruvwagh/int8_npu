// ============================================================================
// File: npu_sram.sv
// Description: Parameterized SRAM Macro Wrapper
//              - Under SYNTHESIS: Blackboxed macro with (* blackbox *) to
//                prevent flip-flop inference traps (keeps area honest).
//              - Under SIMULATION: Behavioral memory model with synchronous
//                1-cycle read latency matching silicon SRAM macros.
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_SRAM_SV
`define NPU_SRAM_SV

`timescale 1ns/1ps

(* blackbox *)
module npu_sram #(
  parameter int DW    = 128,            // Data width (128 bits = 16 bytes)
  parameter int WORDS = 128,            // Depth (128 words = 2 KB per macro)
  parameter int AW    = $clog2(WORDS)   // Address width
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

`ifdef SYNTHESIS
  // Blackboxed macro: area and timing characterised from OpenRAM macro datasheet.
  // No flip-flops are inferred by Yosys.
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
