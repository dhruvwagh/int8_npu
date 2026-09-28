// ============================================================================
// File: npu_fetch.sv
// Description: Tensor Memory Wide Fetch Engine (128-bit port)
//              - Fetches activation/weight tiles into ping-pong buffer banks
//              - Pipelined requests: streams 16 B/cycle without stalling on single-beat latency
//              - Issues fetch_done strobe upon transferring all requested words
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_FETCH_SV
`define NPU_FETCH_SV

`timescale 1ns/1ps

module npu_fetch #(
  parameter int MEM_DW = 128,           // 128 bits = 16 bytes per word
  parameter int ADDR_W = 32,            // Tensor memory address width
  parameter int WORDS  = 128,           // Buffer capacity (words)
  parameter int AW     = $clog2(WORDS)  // 7-bit buffer address
) (
  input  logic                clk,
  input  logic                rst_n,

  // Control Interface (from Sequencer)
  input  logic                fetch_start,     // Pulse to start tile transfer
  input  logic [ADDR_W-1:0]   fetch_base_addr, // Base byte address in tensor memory
  input  logic [7:0]          fetch_words,     // Total 128-bit words to transfer (<= WORDS)
  output logic                fetch_busy,      // Transfer in progress
  output logic                fetch_done,      // Pulse when full tile has been written

  // Tensor Memory Read Bus (Master Interface)
  output logic                mem_req,         // Memory read request
  output logic [ADDR_W-1:0]   mem_addr,        // Memory byte address
  input  logic                mem_ready,       // Memory accepts read request
  input  logic                mem_valid,       // Memory response valid
  input  logic [MEM_DW-1:0]   mem_rdata,       // Memory read data

  // Buffer Write Interface (to idle ping-pong bank)
  output logic                buf_wr_en,
  output logic [AW-1:0]       buf_wr_addr,
  output logic [MEM_DW-1:0]   buf_wr_data
);

  typedef enum logic [1:0] {
    FETCH_IDLE,
    FETCH_RUN
  } state_t;

  state_t state;

  logic [ADDR_W-1:0] base_reg;
  logic [7:0]        total_words_reg;
  logic [7:0]        req_cnt;
  logic [7:0]        resp_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state           <= FETCH_IDLE;
      base_reg        <= '0;
      total_words_reg <= '0;
      req_cnt         <= '0;
      resp_cnt        <= '0;
      fetch_done      <= 1'b0;
      buf_wr_en       <= 1'b0;
      buf_wr_addr     <= '0;
      buf_wr_data     <= '0;
    end else begin
      fetch_done <= 1'b0;
      buf_wr_en  <= 1'b0;

      case (state)
        FETCH_IDLE: begin
          if (fetch_start && (fetch_words > 0)) begin
            state           <= FETCH_RUN;
            base_reg        <= fetch_base_addr;
            total_words_reg <= fetch_words;
            req_cnt         <= '0;
            resp_cnt        <= '0;
          end
        end

        FETCH_RUN: begin
          // Issue read requests until all words have been requested
          if (mem_req && mem_ready) begin
            req_cnt <= req_cnt + 1'b1;
          end

          // Capture incoming memory data beats and write to buffer
          if (mem_valid) begin
            buf_wr_en   <= 1'b1;
            buf_wr_addr <= resp_cnt[AW-1:0];
            buf_wr_data <= mem_rdata;
            resp_cnt    <= resp_cnt + 1'b1;

            if (resp_cnt + 1'b1 == total_words_reg) begin
              state      <= FETCH_IDLE;
              fetch_done <= 1'b1;
            end
          end
        end

        default: state <= FETCH_IDLE;
      endcase
    end
  end

  // Combinational Request Generation
  always_comb begin
    if (state == FETCH_RUN && (req_cnt < total_words_reg)) begin
      mem_req  = 1'b1;
      // 16 bytes (1 << 4) per 128-bit word
      mem_addr = base_reg + (ADDR_W'(req_cnt) << 4);
    end else begin
      mem_req  = 1'b0;
      mem_addr = '0;
    end
  end

  assign fetch_busy = (state != FETCH_IDLE) | fetch_start;

endmodule : npu_fetch

`endif // NPU_FETCH_SV
