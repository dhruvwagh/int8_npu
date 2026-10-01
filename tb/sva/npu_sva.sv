// ============================================================================
// File: npu_sva.sv
// Description: Assertions and functional coverage for npu_top
//              Bound into npu_top (see npu_sva_bind.sv); reads internal signals
//              by upward hierarchical reference. Verilator only (--assert).
//
//              Assertions: invariants the pipelined sequencer and output path
//              rely on, plus AXI4-Lite response-channel stability.
//              Coverage: counters for the corner cases the regression must
//              reach; printed at the end of simulation as
//                "COVER <name> <count>"
//              and checked for zeros by scripts/run_regression.py.
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module npu_sva #(
  parameter int ADDR_W = 32,
  parameter int MEM_DW = 128
) (
  input logic              clk,
  input logic              rst_n,
  // npu_top-level signals (connected by name through the bind)
  input logic              drain_start,
  input logic              drain_ready,
  input logic              drain_inflight,
  input logic              wbq_valid,
  input logic              obuf_swap,
  input logic              wb_start,
  input logic              wb_busy,
  input logic              mem_req,
  input logic              mem_ready,
  input logic [ADDR_W-1:0] mem_addr,
  input logic              bias_wr_en,
  input logic              bias_wr_slot,
  input logic              s_axil_bvalid,
  input logic              s_axil_bready,
  input logic [1:0]        s_axil_bresp,
  input logic              s_axil_rvalid,
  input logic              s_axil_rready,
  input logic [31:0]       s_axil_rdata,
  input logic [1:0]        s_axil_rresp
);

  localparam int BS = $clog2(MEM_DW / 8);

  // Compute-side state encoding in npu_sequencer (cstate_t)
  localparam logic [2:0] C_WAIT  = 3'd2;
  localparam logic [2:0] C_WAVE  = 3'd4;

  default clocking cb @(posedge clk); endclocking
  default disable iff (!rst_n);

  // --------------------------------------------------------------------------
  // Datapath / control invariants
  // --------------------------------------------------------------------------
  // A bank becomes the read bank only after its fetch has fully landed
  a_act_swap_after_fetch: assert property (u_seq.act_buf_swap |-> !u_fetch_act.fetch_busy);
  a_wgt_swap_after_fetch: assert property (u_seq.wgt_buf_swap |-> !u_fetch_wgt.fetch_busy);

  // Both operand buffers are always read in lockstep
  a_reads_lockstep: assert property (u_seq.wgt_rd_en == u_seq.act_rd_en);

  // A clear/capture wave carries zero operands (no read in the same slot)
  a_wave_no_read: assert property (u_seq.clr_acc |-> !u_seq.wgt_rd_en);

  // Capture waves never overwrite shadow registers still being drained: a
  // blocked wave (previous capture's drain not started) is not issued, and
  // consecutive waves are at least CAP_DELAY + 1 cycles apart.
  a_blocked_wave_waits: assert property (
    (u_seq.cstate == C_WAVE) && u_seq.cap_pending |=> !u_seq.clr_acc);

  int unsigned since_wave;
  logic        seen_wave;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      since_wave <= 0;
      seen_wave  <= 1'b0;
    end else if (u_seq.clr_acc && u_seq.cap_pending) begin
      // cap_pending is set in the cycle the capture wave becomes visible
      a_wave_spacing: assert (!seen_wave || since_wave >= u_seq.CAP_DELAY + 1)
        else $error("capture waves %0d cycles apart", since_wave);
      since_wave <= 1;
      seen_wave  <= 1'b1;
    end else begin
      since_wave <= since_wave + 1;
    end
  end

  // Drain never restarts while a drained tile is still in the output path
  a_drain_free: assert property (drain_start |-> !drain_inflight && !wbq_valid);

  // A finished obuf bank never overwrites a queued one
  a_wbq_free: assert property (obuf_swap |-> !wbq_valid);

  // Writeback start only when the engine is idle
  a_wb_idle: assert property (wb_start |-> !wb_busy);

  // Tensor memory accesses are 16-byte aligned
  a_mem_aligned: assert property (mem_req |-> mem_addr[BS-1:0] == '0);

  // Bias slot being written is never the one feeding a drain in progress
  a_bias_slot_safe: assert property (
    bias_wr_en && u_seq.drn_active |-> bias_wr_slot != u_seq.drn_slot);

  // done and err are never raised together
  a_done_xor_err: assert property (!(u_seq.seq_done && u_seq.seq_err));

  // --------------------------------------------------------------------------
  // AXI4-Lite response channels hold until accepted
  // --------------------------------------------------------------------------
  a_axi_b_hold: assert property (s_axil_bvalid && !s_axil_bready |=>
                                 s_axil_bvalid && $stable(s_axil_bresp));
  a_axi_r_hold: assert property (s_axil_rvalid && !s_axil_rready |=>
                                 s_axil_rvalid && $stable(s_axil_rdata) && $stable(s_axil_rresp));

  // --------------------------------------------------------------------------
  // Functional coverage
  // --------------------------------------------------------------------------
  int unsigned cov_ksplice_b2b, cov_wave_blocked, cov_drain_blocked, cov_pop_on_wave,
               cov_compute_starved, cov_bias_slot1, cov_k_inner_order, cov_wbq_while_wb,
               cov_mem_backpressure, cov_tag_fifo_full, cov_wgt_reuse, cov_partial_m,
               cov_partial_n, cov_layer_done, cov_layer_err;

  always @(posedge clk) begin
    if (rst_n) begin
      // next K slice of the same output block streams with no gap
      if (u_seq.c_last_read && !u_seq.c_tile.last_k && u_seq.mbox_valid) cov_ksplice_b2b++;
      // capture wave waits for the previous drain to start
      if (u_seq.cstate == C_WAVE && u_seq.cap_pending) cov_wave_blocked++;
      // drain ready to start but output path busy
      if (u_seq.cap_ready && !drain_ready) cov_drain_blocked++;
      // next tile popped in the same cycle as a capture wave
      if (u_seq.cstate == C_WAVE && u_seq.c_pop) cov_pop_on_wave++;
      // compute idle waiting for the fetch side
      if (u_seq.cstate == C_WAIT && !u_seq.mbox_valid) cov_compute_starved++;
      if (u_seq.fetch_bias_start && u_seq.fetch_bias_slot) cov_bias_slot1++;
      if (u_seq.fetch_act_start && u_seq.k_split_multitile) cov_k_inner_order++;
      if (wbq_valid && wb_busy) cov_wbq_while_wb++;
      if (mem_req && !mem_ready) cov_mem_backpressure++;
      if (u_arb.tag_full) cov_tag_fifo_full++;
      // a tile issued without refetching weights (residency)
      if (u_seq.fetch_act_start && !u_seq.fetch_wgt_start) cov_wgt_reuse++;
      if (u_seq.c_pop && u_seq.mbox.m_len != 8'(u_seq.ARR_M)) cov_partial_m++;
      if (u_seq.c_pop && u_seq.mbox.n_len != 8'(u_seq.ARR_N)) cov_partial_n++;
      if (u_seq.seq_done) cov_layer_done++;
      if (u_seq.seq_err)  cov_layer_err++;
    end
  end

  final begin
    $display("COVER ksplice_back_to_back   %0d", cov_ksplice_b2b);
    $display("COVER wave_blocked_by_drain  %0d", cov_wave_blocked);
    $display("COVER drain_blocked_by_output %0d", cov_drain_blocked);
    $display("COVER pop_on_wave_cycle      %0d", cov_pop_on_wave);
    $display("COVER compute_starved        %0d", cov_compute_starved);
    $display("COVER bias_slot1             %0d", cov_bias_slot1);
    $display("COVER k_inner_loop_order     %0d", cov_k_inner_order);
    $display("COVER wbq_full_during_wb     %0d", cov_wbq_while_wb);
    $display("COVER mem_backpressure       %0d", cov_mem_backpressure);
    $display("COVER tag_fifo_full          %0d", cov_tag_fifo_full);
    $display("COVER weight_reuse           %0d", cov_wgt_reuse);
    $display("COVER partial_m_tile         %0d", cov_partial_m);
    $display("COVER partial_n_tile         %0d", cov_partial_n);
    $display("COVER layer_done             %0d", cov_layer_done);
    $display("COVER layer_error            %0d", cov_layer_err);
  end

endmodule : npu_sva
