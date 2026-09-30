// ============================================================================
// File: npu_buf_wgt.sv
// Description: Ping-Pong Weight Buffer wrapping two npu_sram macros (4 KB total)
//              - Bank A (Ping) reads weights for current compute tile
//              - Bank B (Pong) concurrently writes incoming tile from memory
//              - Supports Weight-Tile Residency: weight tile held resident
//                across multiple activation sweeps without swapping or refetching
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_BUF_WGT_SV
`define NPU_BUF_WGT_SV

`timescale 1ns/1ps

module npu_buf_wgt #(
  parameter int DW    = 128,            // 128 bits = 16 weight bytes
  parameter int WORDS = 128,            // 128 words = 2 KB per bank (4 KB total)
  parameter int AW    = $clog2(WORDS)   // 7-bit address
) (
  input  logic          clk,
  input  logic          rst_n,

  // Bank swap control
  input  logic          swap,           // Strobe to swap read/write banks
  output logic          cur_bank,       // Current read bank index (0 or 1)

  // Compute Read Interface (feeds systolic array skew registers)
  input  logic          rd_en,
  input  logic [AW-1:0] rd_addr,
  output logic [DW-1:0] rd_data,

  // Fetch Write Interface (driven by tensor memory fetch unit)
  input  logic          wr_en,
  input  logic [AW-1:0] wr_addr,
  input  logic [DW-1:0] wr_data
);

  // Active bank tracking register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cur_bank <= 1'b0;
    end else if (swap) begin
      cur_bank <= ~cur_bank;
    end
  end

  // Bank 0 Control Signals
  logic          b0_we, b0_re;
  logic [AW-1:0] b0_waddr, b0_raddr;
  logic [DW-1:0] b0_wdata, b0_rdata;

  // Bank 1 Control Signals
  logic          b1_we, b1_re;
  logic [AW-1:0] b1_waddr, b1_raddr;
  logic [DW-1:0] b1_wdata, b1_rdata;

  // Muxing between compute and fetch ports
  always_comb begin
    if (cur_bank == 1'b0) begin
      // Bank 0 is Compute Read
      b0_re    = rd_en;
      b0_raddr = rd_addr;
      b0_we    = 1'b0;
      b0_waddr = '0;
      b0_wdata = '0;

      // Bank 1 is Fetch Write
      b1_we    = wr_en;
      b1_waddr = wr_addr;
      b1_wdata = wr_data;
      b1_re    = 1'b0;
      b1_raddr = '0;

    end else begin
      // Bank 1 is Compute Read
      b1_re    = rd_en;
      b1_raddr = rd_addr;
      b1_we    = 1'b0;
      b1_waddr = '0;
      b1_wdata = '0;

      // Bank 0 is Fetch Write
      b0_we    = wr_en;
      b0_waddr = wr_addr;
      b0_wdata = wr_data;
      b0_re    = 1'b0;
      b0_raddr = '0;

    end
  end

  // Read data comes back one cycle after the read, so select it with the bank
  // that was read, not the current bank. This keeps a swap issued right after
  // the last read of a tile from returning the other bank's data.
  logic rd_bank_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_bank_q <= 1'b0;
    end else if (rd_en) begin
      rd_bank_q <= cur_bank;
    end
  end
  assign rd_data = rd_bank_q ? b1_rdata : b0_rdata;

  // --------------------------------------------------------------------------
  // SRAM Bank Instantiations
  // --------------------------------------------------------------------------
  npu_sram #(
    .DW    (DW),
    .WORDS (WORDS),
    .AW    (AW)
  ) u_sram_bank0 (
    .clk   (clk),
    .csb   (1'b0),
    .we    (b0_we),
    .waddr (b0_waddr),
    .wdata (b0_wdata),
    .re    (b0_re),
    .raddr (b0_raddr),
    .rdata (b0_rdata)
  );

  npu_sram #(
    .DW    (DW),
    .WORDS (WORDS),
    .AW    (AW)
  ) u_sram_bank1 (
    .clk   (clk),
    .csb   (1'b0),
    .we    (b1_we),
    .waddr (b1_waddr),
    .wdata (b1_wdata),
    .re    (b1_re),
    .raddr (b1_raddr),
    .rdata (b1_rdata)
  );

endmodule : npu_buf_wgt

`endif // NPU_BUF_WGT_SV
