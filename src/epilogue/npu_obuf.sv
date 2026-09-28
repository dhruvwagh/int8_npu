// ============================================================================
// File: npu_obuf.sv
// Description: Double-Buffered Output Staging Buffer (512 B total)
//              - Wraps two npu_sram macros (each 16 words x 128 bits = 256 B)
//              - Bank A: Written by npu_epilogue during column drain (16 cycles)
//              - Bank B: Concurrently read by DMA writeback engine to tensor memory
//              - Instantaneous zero-bubble bank swap on `swap` strobe
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_OBUF_SV
`define NPU_OBUF_SV

`timescale 1ns/1ps

module npu_obuf #(
  parameter int DW    = 128,           // 128 bits = 16 bytes per column word
  parameter int WORDS = 16,            // 16 words = 256 B per bank (512 B total)
  parameter int AW    = $clog2(WORDS)  // 4 bits
) (
  input  logic          clk,
  input  logic          rst_n,

  // Bank Swap Control
  input  logic          swap,          // Pulse to swap write/read banks
  output logic          cur_bank,      // 0: Bank 1 is write, Bank 0 is read
                                       // 1: Bank 0 is write, Bank 1 is read

  // Epilogue Write Port (from npu_epilogue)
  input  logic          wr_en,
  input  logic [AW-1:0] wr_addr,
  input  logic [DW-1:0] wr_data,

  // DMA Read Port (to tensor memory writeback)
  input  logic          rd_en,
  input  logic [AW-1:0] rd_addr,
  output logic [DW-1:0] rd_data
);

  // Active bank register
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

  // Muxing between epilogue write and DMA read ports
  always_comb begin
    if (cur_bank == 1'b0) begin
      // Bank 0 is DMA Read
      b0_re    = rd_en;
      b0_raddr = rd_addr;
      b0_we    = 1'b0;
      b0_waddr = '0;
      b0_wdata = '0;

      // Bank 1 is Epilogue Write
      b1_we    = wr_en;
      b1_waddr = wr_addr;
      b1_wdata = wr_data;
      b1_re    = 1'b0;
      b1_raddr = '0;

      rd_data  = b0_rdata;
    end else begin
      // Bank 1 is DMA Read
      b1_re    = rd_en;
      b1_raddr = rd_addr;
      b1_we    = 1'b0;
      b1_waddr = '0;
      b1_wdata = '0;

      // Bank 0 is Epilogue Write
      b0_we    = wr_en;
      b0_waddr = wr_addr;
      b0_wdata = wr_data;
      b0_re    = 1'b0;
      b0_raddr = '0;

      rd_data  = b1_rdata;
    end
  end

  // --------------------------------------------------------------------------
  // SRAM Bank Instantiations
  // --------------------------------------------------------------------------
  npu_sram #(
    .DW    (DW),
    .WORDS (WORDS),
    .AW    (AW)
  ) u_obuf_bank0 (
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
  ) u_obuf_bank1 (
    .clk   (clk),
    .csb   (1'b0),
    .we    (b1_we),
    .waddr (b1_waddr),
    .wdata (b1_wdata),
    .re    (b1_re),
    .raddr (b1_raddr),
    .rdata (b1_rdata)
  );

endmodule : npu_obuf

`endif // NPU_OBUF_SV
