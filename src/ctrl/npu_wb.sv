// ============================================================================
// File: npu_wb.sv
// Description: Output Writeback Engine
//              - Streams one drained tile (n_len words) from the read bank of
//                npu_obuf to tensor memory at 1 word/cycle
//              - obuf reads have 1-cycle latency, so reads run ahead of writes
//                into a 2-entry skid FIFO; memory backpressure (mem_ready = 0)
//                stalls reads without dropping data
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_WB_SV
`define NPU_WB_SV

`timescale 1ns/1ps

module npu_wb #(
  parameter int MEM_DW = 128,
  parameter int ADDR_W = 32,
  parameter int WORDS  = 16,             // obuf bank depth (= ARR_N)
  parameter int AW     = $clog2(WORDS)
) (
  input  logic                clk,
  input  logic                rst_n,

  // Control
  input  logic                wb_start,       // Pulse: read bank now holds a full tile
  input  logic [ADDR_W-1:0]   wb_base_addr,   // Byte address of output word 0
  input  logic [7:0]          wb_words,       // Valid columns (1..WORDS)
  output logic                wb_busy,

  // obuf Read Port
  output logic                obuf_rd_en,
  output logic [AW-1:0]       obuf_rd_addr,
  input  logic [MEM_DW-1:0]   obuf_rd_data,

  // Tensor Memory Write Port
  output logic                mem_req,
  output logic [ADDR_W-1:0]   mem_addr,
  output logic [MEM_DW-1:0]   mem_wdata,
  input  logic                mem_ready
);

  logic              active;
  logic [ADDR_W-1:0] base_reg;
  logic [7:0]        total_reg;
  logic [7:0]        rd_cnt;          // obuf reads issued
  logic [7:0]        wr_cnt;          // memory writes accepted
  logic              rd_inflight;     // read issued last cycle, data arrives now

  // 2-entry skid FIFO between obuf read data and memory write port
  logic [MEM_DW-1:0] fifo_q [2];
  logic              fifo_rd_ptr, fifo_wr_ptr;
  logic [1:0]        fifo_cnt;

  wire pop  = mem_req && mem_ready;
  wire push = rd_inflight;

  // Issue a read only if its data is guaranteed a FIFO slot when it lands
  wire [1:0] occupancy_next = fifo_cnt + 2'(rd_inflight) - 2'(pop);
  assign obuf_rd_en   = active && (rd_cnt < total_reg) && (occupancy_next < 2'd2);
  assign obuf_rd_addr = rd_cnt[AW-1:0];

  assign mem_req   = active && (fifo_cnt != 0);
  localparam int WORD_SHIFT = $clog2(MEM_DW / 8);     // bytes per memory word
  assign mem_addr  = base_reg + (ADDR_W'(wr_cnt) << WORD_SHIFT);
  assign mem_wdata = fifo_q[fifo_rd_ptr];

  assign wb_busy = active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active      <= 1'b0;
      base_reg    <= '0;
      total_reg   <= '0;
      rd_cnt      <= '0;
      wr_cnt      <= '0;
      rd_inflight <= 1'b0;
      fifo_rd_ptr <= 1'b0;
      fifo_wr_ptr <= 1'b0;
      fifo_cnt    <= '0;
      fifo_q[0]   <= '0;
      fifo_q[1]   <= '0;
    end else begin
      rd_inflight <= obuf_rd_en;

      if (obuf_rd_en) begin
        rd_cnt <= rd_cnt + 1'b1;
      end

      if (push) begin
        fifo_q[fifo_wr_ptr] <= obuf_rd_data;
        fifo_wr_ptr         <= ~fifo_wr_ptr;
      end
      if (pop) begin
        fifo_rd_ptr <= ~fifo_rd_ptr;
        wr_cnt      <= wr_cnt + 1'b1;
      end
      fifo_cnt <= fifo_cnt + 2'(push) - 2'(pop);

      if (!active) begin
        if (wb_start && (wb_words != 0)) begin
          active    <= 1'b1;
          base_reg  <= wb_base_addr;
          total_reg <= wb_words;
          rd_cnt    <= '0;
          wr_cnt    <= '0;
        end
      end else if (pop && (wr_cnt + 1'b1 == total_reg)) begin
        active <= 1'b0;
      end
    end
  end

endmodule : npu_wb

`endif // NPU_WB_SV
