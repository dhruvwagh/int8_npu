// ============================================================================
// File: npu_mem_arb.sv
// Description: Tensor Memory Port Arbiter
//              - Shares one MEM_DW-wide memory port between NREQ requesters
//                (writeback, weight fetch, activation fetch, bias fetch)
//              - Fixed priority: requester 0 highest
//              - Memory returns read data in order; a tag FIFO records which
//                requester owns each outstanding read so responses are routed
//                back to it. Reads stall when MAX_OUTSTANDING are in flight.
//              - All NPU traffic shares this port, so its width is the real
//                bandwidth limit seen by the whole accelerator.
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_MEM_ARB_SV
`define NPU_MEM_ARB_SV

`timescale 1ns/1ps

module npu_mem_arb #(
  parameter int NREQ            = 4,
  parameter int MEM_DW          = 128,
  parameter int ADDR_W          = 32,
  parameter int MAX_OUTSTANDING = 8
) (
  input  logic                     clk,
  input  logic                     rst_n,

  // Requester side
  input  logic [NREQ-1:0]          req,
  input  logic [NREQ-1:0]          req_we,
  input  logic [NREQ*ADDR_W-1:0]   req_addr,
  input  logic [NREQ*MEM_DW-1:0]   req_wdata,
  output logic [NREQ-1:0]          req_ready,
  output logic [NREQ-1:0]          rsp_valid,
  output logic [MEM_DW-1:0]        rsp_rdata,

  // Memory side (in-order read responses)
  output logic                     mem_req,
  output logic                     mem_we,
  output logic [ADDR_W-1:0]        mem_addr,
  output logic [MEM_DW-1:0]        mem_wdata,
  input  logic                     mem_ready,
  input  logic                     mem_rvalid,
  input  logic [MEM_DW-1:0]        mem_rdata
);

  localparam int IDW = (NREQ > 1) ? $clog2(NREQ) : 1;
  localparam int PW  = $clog2(MAX_OUTSTANDING);

  // --------------------------------------------------------------------------
  // Outstanding-read tag FIFO
  // --------------------------------------------------------------------------
  logic [IDW-1:0] tag_q [MAX_OUTSTANDING];
  logic [PW-1:0]  tag_wr_ptr, tag_rd_ptr;
  logic [PW:0]    tag_cnt;
  wire            tag_full = (tag_cnt == (PW+1)'(MAX_OUTSTANDING));

  // --------------------------------------------------------------------------
  // Fixed-priority grant; reads are ineligible while the tag FIFO is full
  // --------------------------------------------------------------------------
  logic [NREQ-1:0] eligible;
  logic            grant_any;
  logic [IDW-1:0]  grant_id;

  always_comb begin
    for (int i = 0; i < NREQ; i++) begin
      eligible[i] = req[i] && (req_we[i] || !tag_full);
    end
    grant_any = 1'b0;
    grant_id  = '0;
    for (int i = NREQ-1; i >= 0; i--) begin
      if (eligible[i]) begin
        grant_any = 1'b1;
        grant_id  = IDW'(i);
      end
    end
  end

  assign mem_req   = grant_any;
  assign mem_we    = req_we[grant_id];
  assign mem_addr  = req_addr[grant_id*ADDR_W +: ADDR_W];
  assign mem_wdata = req_wdata[grant_id*MEM_DW +: MEM_DW];

  always_comb begin
    req_ready = '0;
    if (grant_any) begin
      req_ready[grant_id] = mem_ready;
    end
  end

  wire tag_push = grant_any && mem_ready && !mem_we;
  wire tag_pop  = mem_rvalid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tag_wr_ptr <= '0;
      tag_rd_ptr <= '0;
      tag_cnt    <= '0;
      for (int i = 0; i < MAX_OUTSTANDING; i++) begin
        tag_q[i] <= '0;
      end
    end else begin
      if (tag_push) begin
        tag_q[tag_wr_ptr] <= grant_id;
        tag_wr_ptr        <= tag_wr_ptr + 1'b1;
      end
      if (tag_pop) begin
        tag_rd_ptr <= tag_rd_ptr + 1'b1;
      end
      tag_cnt <= tag_cnt + (PW+1)'(tag_push) - (PW+1)'(tag_pop);
    end
  end

  // --------------------------------------------------------------------------
  // Response routing
  // --------------------------------------------------------------------------
  always_comb begin
    rsp_valid = '0;
    if (mem_rvalid) begin
      rsp_valid[tag_q[tag_rd_ptr]] = 1'b1;
    end
  end
  assign rsp_rdata = mem_rdata;

endmodule : npu_mem_arb

`endif // NPU_MEM_ARB_SV
