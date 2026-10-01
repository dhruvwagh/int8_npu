// ============================================================================
// File: npu_sequencer.sv
// Description: Pipelined Multi-Loop Tiling Sequencer
//              Two decoupled controllers joined by a one-entry tile mailbox:
//
//              FETCH side  walks the tile loop nest (M -> K -> N for weight
//                          residency, or K innermost when K is split and N > 16),
//                          fetches the next tile's activations (and weights /
//                          bias when they change) into the idle ping-pong banks,
//                          then posts a tile descriptor. It runs one tile ahead
//                          of compute, which is what the ping-pong buffers are for.
//
//              COMPUTE side takes a descriptor, swaps banks, and streams k_len
//                          buffer words into the array. Tiles of the same output
//                          block (K split) stream back to back with no bubble.
//                          After a tile that completes an output block it issues
//                          a one-cycle "capture wave": a clear pulse with zero
//                          operands that sweeps the array diagonally, copying each
//                          PE's result into the drain shadow registers and
//                          clearing it, so the next tile can follow immediately.
//
//              DRAIN scheduling: drain_start fires ARR_M+3 cycles after a wave is
//                          issued (column 0 of the shadow is then complete) once
//                          the output path is free. A new wave waits until the
//                          previous drain has started, or it would overwrite the
//                          shadow registers being read.
//
//              - Hardware performance counters: total cycles, MACs, stall cycles
//
// Tensor memory layout (packed by the host/compiler, see docs/SPEC.md §9):
//   A (weights, M x K):     word (m0/16, k) at a_base + m0*K + k*16
//   B (activations, K x N): word (n0/16, k) at b_base + n0*K + k*16
//   Bias (int32 per M row): 4 words per M panel at bias_base + m0*4
//   C (outputs, M x N):     word (m0/16, n) at c_base + m0*N + n*16
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_SEQUENCER_SV
`define NPU_SEQUENCER_SV

`timescale 1ns/1ps

module npu_sequencer #(
  parameter int ARR_M      = 16,
  parameter int ARR_N      = 16,
  parameter int K_TILE_MAX = 128,           // Maximum K depth per sub-reduction tile
  parameter int ADDR_W     = 32,
  parameter int DATA_W     = 8
) (
  input  logic                clk,
  input  logic                rst_n,

  // CSR Layer Configuration & Control
  input  logic                seq_start,       // Strobe to begin layer execution
  input  logic                seq_abort,       // Abort execution
  input  logic [15:0]         dim_m,           // Total output channels M
  input  logic [15:0]         dim_n,           // Total output spatial points N
  input  logic [15:0]         dim_k,           // Total reduction depth K
  input  logic [ADDR_W-1:0]   a_base,          // Weights base address
  input  logic [ADDR_W-1:0]   b_base,          // Activations base address
  input  logic [ADDR_W-1:0]   c_base,          // Output matrix base address
  input  logic [ADDR_W-1:0]   bias_base,       // Bias vector base address

  // Status & Performance Counters
  output logic                seq_busy,        // Sequencer is executing
  output logic                seq_done,        // 1-cycle pulse upon layer completion
  output logic                seq_err,         // Error flag (e.g. 0 dimensions)
  output logic [31:0]         perf_cycles,     // Total cycles spent on layer
  output logic [31:0]         perf_macs,       // Total active MAC operations executed
  output logic [31:0]         perf_stall,      // Compute-side cycles waiting for data or drain
  input  logic                perf_clr,        // Clear perf counters (when idle)

  // Weight Fetch Engine Interface
  output logic                fetch_wgt_start,
  output logic [ADDR_W-1:0]   fetch_wgt_base_addr,
  output logic [7:0]          fetch_wgt_words,
  input  logic                fetch_wgt_done,

  // Activation Fetch Engine Interface
  output logic                fetch_act_start,
  output logic [ADDR_W-1:0]   fetch_act_base_addr,
  output logic [7:0]          fetch_act_words,
  input  logic                fetch_act_done,

  // Bias Fetch Engine Interface (4 words per M panel, into one of two slots)
  output logic                fetch_bias_start,
  output logic [ADDR_W-1:0]   fetch_bias_base_addr,
  output logic                fetch_bias_slot,
  input  logic                fetch_bias_done,

  // Ping-Pong Buffer Control
  output logic                wgt_buf_swap,
  output logic                act_buf_swap,

  // Buffer Compute Read Address Interface
  output logic                wgt_rd_en,
  output logic [$clog2(K_TILE_MAX)-1:0] wgt_rd_addr,
  output logic                act_rd_en,
  output logic [$clog2(K_TILE_MAX)-1:0] act_rd_addr,
  output logic [ARR_M-1:0]    active_m_mask,   // valid with rd_en: 1 for active rows
  output logic [ARR_N-1:0]    active_n_mask,   // valid with rd_en: 1 for active cols

  // Array control
  output logic                clr_acc,         // Clear/capture wave (issued with rd_en = 0)

  // Drain / output path
  output logic                drain_start,
  output logic [ADDR_W-1:0]   drain_c_addr,    // valid with drain_start
  output logic [7:0]          drain_n_len,     // valid with drain_start
  output logic                drain_bias_slot, // valid with drain_start
  input  logic                drain_done,
  input  logic                drain_ready,     // output path can take another tile
  input  logic                out_idle         // nothing left in drain/epilogue/writeback
);

  localparam int KAW    = $clog2(K_TILE_MAX);
  localparam int WORD_B = ARR_M;                // Bytes per 128-bit panel word
  // A wave issued in cycle x is visible at x+1, aligned with the buffer read
  // data at x+2, registered in the operand stage, and reaches PE(0,0) at x+3
  // and PE(ARR_M-1,0) at x+ARR_M+2, where the shadow register captures at the
  // end of the cycle. drain_start must be visible at x+ARR_M+3 or later: the
  // timer below fires at x+ARR_M+2.
  localparam int OPERAND_LAT = 2;              // SRAM read + operand register
  localparam int CAP_DELAY   = ARR_M + OPERAND_LAT;

  initial begin
    if (ARR_M != ARR_N) $fatal(1, "npu_sequencer: panel layout assumes ARR_M == ARR_N");
  end

  // --------------------------------------------------------------------------
  // Latched layer configuration
  // --------------------------------------------------------------------------
  logic [15:0]       reg_m, reg_n, reg_k;
  logic [ADDR_W-1:0] reg_a_base, reg_b_base, reg_c_base, reg_bias_base;
  logic              k_split_multitile;   // K innermost (K > K_TILE and N > ARR_N)

  // --------------------------------------------------------------------------
  // Tile descriptor passed from fetch to compute
  // --------------------------------------------------------------------------
  typedef struct packed {
    logic [7:0]        m_len;
    logic [7:0]        n_len;
    logic [7:0]        k_len;
    logic              last_k;       // tile completes an output block -> capture wave
    logic              new_wgt;      // this tile's weights wait in the idle bank
    logic              last_tile;    // final tile of the layer
    logic [ADDR_W-1:0] c_addr;       // output address of this block
    logic              bias_slot;
    logic              rel_slot;     // last block of its M panel: free the bias slot after drain
  } tile_t;

  tile_t mbox;          // mailbox: fetched, waiting for compute
  logic  mbox_valid;

  // Thermometer mask with the low `len` bits set
  function automatic logic [ARR_M-1:0] mask_of(input logic [7:0] len);
    for (int i = 0; i < ARR_M; i++) mask_of[i] = (8'(i) < len);
  endfunction

  // ==========================================================================
  // FETCH side state
  // ==========================================================================
  typedef enum logic [1:0] { F_IDLE, F_ISSUE, F_WAIT, F_DONE } fstate_t;
  fstate_t fstate;

  logic [15:0] m0, n0, k0;
  logic        panel_slot;          // bias slot of the current M panel
  logic        wgt_full;            // fetched weights wait in the idle bank
  logic [1:0]  bias_busy;           // slot holds a panel that is not fully drained
  logic        f_wgt_pend, f_act_pend, f_bias_pend;
  tile_t       f_tile;

  // Key of the weight tile most recently fetched (weights change on (m0, k0))
  logic [15:0] wgt_key_m, wgt_key_k;
  logic        wgt_key_valid;

  // Geometry of the tile at the current loop position
  logic [15:0] rem_m, rem_n, rem_k;
  logic [7:0]  t_m_len, t_n_len, t_k_len;
  logic        t_last_m, t_last_n, t_last_kt, t_need_wgt, t_need_bias, t_last_tile;

  always_comb begin
    rem_m = reg_m - m0;
    rem_n = reg_n - n0;
    rem_k = reg_k - k0;
    t_m_len     = (rem_m < 16'(ARR_M))      ? 8'(rem_m) : 8'(ARR_M);
    t_n_len     = (rem_n < 16'(ARR_N))      ? 8'(rem_n) : 8'(ARR_N);
    t_k_len     = (rem_k < 16'(K_TILE_MAX)) ? 8'(rem_k) : 8'(K_TILE_MAX);
    t_last_m    = (32'(m0) + ARR_M)      >= 32'(reg_m);
    t_last_n    = (32'(n0) + ARR_N)      >= 32'(reg_n);
    t_last_kt   = (32'(k0) + K_TILE_MAX) >= 32'(reg_k);
    t_need_wgt  = !(wgt_key_valid && wgt_key_m == m0 && wgt_key_k == k0);
    t_need_bias = (k0 == 0) && (n0 == 0);
    t_last_tile = t_last_m && t_last_n && t_last_kt;
  end

  // Issue the next tile's fetches only into free banks / slots
  wire f_can_issue = !mbox_valid
                   && !(t_need_wgt && wgt_full)
                   && !(t_need_bias && bias_busy[panel_slot]);

  // ==========================================================================
  // COMPUTE side state
  // ==========================================================================
  typedef enum logic [2:0] { C_IDLE, C_CLEAR, C_WAIT, C_STREAM, C_WAVE, C_FINISH } cstate_t;
  cstate_t cstate;

  tile_t      c_tile;               // tile being streamed
  logic [7:0] c_cnt;                // next read address

  // Drain scheduling
  logic              cap_pending;   // wave issued, its drain not yet started
  logic [7:0]        cap_timer;
  logic [ADDR_W-1:0] cap_c_addr;
  logic [7:0]        cap_n_len;
  logic              cap_slot, cap_rel;
  logic              drn_active, drn_slot, drn_rel;

  wire cap_ready = cap_pending && (cap_timer == 8'd1);
  wire wave_ok   = !cap_pending;

  wire c_last_read = (cstate == C_STREAM) && (c_cnt + 1'b1 == c_tile.k_len);

  // Pop the mailbox and start the next tile. Popping in the cycle of the last
  // read (same output block) or of the wave keeps tiles back to back.
  wire c_pop = mbox_valid && (
                 (cstate == C_WAIT) ||
                 (c_last_read && !c_tile.last_k) ||
                 ((cstate == C_WAVE) && wave_ok && !c_tile.last_tile));

  // --------------------------------------------------------------------------
  // Sequential logic
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      reg_m <= '0; reg_n <= '0; reg_k <= '0;
      reg_a_base <= '0; reg_b_base <= '0; reg_c_base <= '0; reg_bias_base <= '0;
      k_split_multitile <= 1'b0;
      seq_done <= 1'b0; seq_err <= 1'b0;
      perf_cycles <= '0; perf_macs <= '0; perf_stall <= '0;

      fstate <= F_IDLE;
      m0 <= '0; n0 <= '0; k0 <= '0;
      panel_slot <= 1'b0;
      wgt_full <= 1'b0; bias_busy <= '0;
      f_wgt_pend <= 1'b0; f_act_pend <= 1'b0; f_bias_pend <= 1'b0;
      f_tile <= '0;
      wgt_key_m <= '0; wgt_key_k <= '0; wgt_key_valid <= 1'b0;
      mbox <= '0; mbox_valid <= 1'b0;
      fetch_wgt_start <= 1'b0; fetch_wgt_base_addr <= '0; fetch_wgt_words <= '0;
      fetch_act_start <= 1'b0; fetch_act_base_addr <= '0; fetch_act_words <= '0;
      fetch_bias_start <= 1'b0; fetch_bias_base_addr <= '0; fetch_bias_slot <= 1'b0;

      cstate <= C_IDLE;
      c_tile <= '0; c_cnt <= '0;
      wgt_buf_swap <= 1'b0; act_buf_swap <= 1'b0;
      wgt_rd_en <= 1'b0; wgt_rd_addr <= '0; act_rd_en <= 1'b0; act_rd_addr <= '0;
      active_m_mask <= '0; active_n_mask <= '0;
      clr_acc <= 1'b0;

      cap_pending <= 1'b0; cap_timer <= '0;
      cap_c_addr <= '0; cap_n_len <= '0; cap_slot <= 1'b0; cap_rel <= 1'b0;
      drain_start <= 1'b0; drain_c_addr <= '0; drain_n_len <= '0; drain_bias_slot <= 1'b0;
      drn_active <= 1'b0; drn_slot <= 1'b0; drn_rel <= 1'b0;
    end else if (seq_abort) begin
      fstate <= F_IDLE; cstate <= C_IDLE;
      mbox_valid <= 1'b0;
      wgt_full <= 1'b0; bias_busy <= '0; wgt_key_valid <= 1'b0;
      f_wgt_pend <= 1'b0; f_act_pend <= 1'b0; f_bias_pend <= 1'b0;
      fetch_wgt_start <= 1'b0; fetch_act_start <= 1'b0; fetch_bias_start <= 1'b0;
      wgt_buf_swap <= 1'b0; act_buf_swap <= 1'b0;
      wgt_rd_en <= 1'b0; act_rd_en <= 1'b0; clr_acc <= 1'b0;
      cap_pending <= 1'b0; drain_start <= 1'b0; drn_active <= 1'b0;
      seq_done <= 1'b0;
    end else begin
      // Single-cycle strobes
      seq_done         <= 1'b0;
      seq_err          <= 1'b0;
      fetch_wgt_start  <= 1'b0;
      fetch_act_start  <= 1'b0;
      fetch_bias_start <= 1'b0;
      wgt_buf_swap     <= 1'b0;
      act_buf_swap     <= 1'b0;
      wgt_rd_en        <= 1'b0;
      act_rd_en        <= 1'b0;
      clr_acc          <= 1'b0;
      drain_start      <= 1'b0;

      if (fstate != F_IDLE || cstate != C_IDLE) perf_cycles <= perf_cycles + 1'b1;

      // ======================================================================
      // FETCH side
      // ======================================================================
      if (fetch_wgt_done)  f_wgt_pend  <= 1'b0;
      if (fetch_act_done)  f_act_pend  <= 1'b0;
      if (fetch_bias_done) f_bias_pend <= 1'b0;

      case (fstate)
        F_IDLE: begin
          if (perf_clr && cstate == C_IDLE) begin
            perf_cycles <= '0; perf_macs <= '0; perf_stall <= '0;
          end
          if (seq_start && cstate == C_IDLE) begin
            if (dim_m == 0 || dim_n == 0 || dim_k == 0) begin
              seq_err <= 1'b1;
            end else begin
              reg_m <= dim_m; reg_n <= dim_n; reg_k <= dim_k;
              reg_a_base <= a_base; reg_b_base <= b_base;
              reg_c_base <= c_base; reg_bias_base <= bias_base;
              k_split_multitile <= (dim_k > 16'(K_TILE_MAX)) && (dim_n > 16'(ARR_N));
              perf_cycles <= '0; perf_macs <= '0; perf_stall <= '0;
              m0 <= '0; n0 <= '0; k0 <= '0;
              panel_slot <= 1'b0;
              wgt_full <= 1'b0; bias_busy <= '0; wgt_key_valid <= 1'b0;
              mbox_valid <= 1'b0;
              fstate <= F_ISSUE;
              cstate <= C_CLEAR;
            end
          end
        end

        F_ISSUE: begin
          if (f_can_issue) begin
            fetch_act_start     <= 1'b1;
            fetch_act_base_addr <= reg_b_base + ADDR_W'(n0) * ADDR_W'(reg_k) + ADDR_W'(k0) * ADDR_W'(WORD_B);
            fetch_act_words     <= t_k_len;
            f_act_pend          <= 1'b1;
            if (t_need_wgt) begin
              fetch_wgt_start     <= 1'b1;
              fetch_wgt_base_addr <= reg_a_base + ADDR_W'(m0) * ADDR_W'(reg_k) + ADDR_W'(k0) * ADDR_W'(WORD_B);
              fetch_wgt_words     <= t_k_len;
              f_wgt_pend          <= 1'b1;
              wgt_full            <= 1'b1;
              wgt_key_m           <= m0;
              wgt_key_k           <= k0;
              wgt_key_valid       <= 1'b1;
            end
            if (t_need_bias) begin
              fetch_bias_start      <= 1'b1;
              fetch_bias_base_addr  <= reg_bias_base + (ADDR_W'(m0) << 2);
              fetch_bias_slot       <= panel_slot;
              f_bias_pend           <= 1'b1;
              bias_busy[panel_slot] <= 1'b1;
            end

            f_tile.m_len     <= t_m_len;
            f_tile.n_len     <= t_n_len;
            f_tile.k_len     <= t_k_len;
            f_tile.last_k    <= t_last_kt;
            f_tile.new_wgt   <= t_need_wgt;
            f_tile.last_tile <= t_last_tile;
            f_tile.c_addr    <= reg_c_base + ADDR_W'(m0) * ADDR_W'(reg_n) + ADDR_W'(n0) * ADDR_W'(WORD_B);
            f_tile.bias_slot <= panel_slot;
            // The last output block of an M panel finishes both its N and K
            // loops (true in either loop order)
            f_tile.rel_slot  <= t_last_n && t_last_kt;

            // Advance the loop nest
            if (k_split_multitile) begin
              if (!t_last_kt) begin
                k0 <= k0 + 16'(K_TILE_MAX);
              end else begin
                k0 <= '0;
                if (!t_last_n) begin
                  n0 <= n0 + 16'(ARR_N);
                end else begin
                  n0 <= '0;
                  m0 <= m0 + 16'(ARR_M);
                  panel_slot <= ~panel_slot;
                end
              end
            end else begin
              if (!t_last_n) begin
                n0 <= n0 + 16'(ARR_N);
              end else begin
                n0 <= '0;
                if (!t_last_kt) begin
                  k0 <= k0 + 16'(K_TILE_MAX);
                end else begin
                  k0 <= '0;
                  m0 <= m0 + 16'(ARR_M);
                  panel_slot <= ~panel_slot;
                end
              end
            end
            fstate <= F_WAIT;
          end
        end

        // Post the descriptor once every fetch for this tile has landed
        F_WAIT: begin
          if (!f_act_pend && !f_wgt_pend && !f_bias_pend) begin
            mbox       <= f_tile;
            mbox_valid <= 1'b1;
            if (f_tile.last_tile) fstate <= F_DONE;
            else                  fstate <= F_ISSUE;
          end
        end

        F_DONE: begin
          if (cstate == C_IDLE) fstate <= F_IDLE;
        end

        default: fstate <= F_IDLE;
      endcase

      // ======================================================================
      // COMPUTE side
      // ======================================================================
      case (cstate)
        C_IDLE: ;

        // Clear the accumulators once at layer start (no capture)
        C_CLEAR: begin
          clr_acc <= 1'b1;
          cstate  <= C_WAIT;
        end

        C_WAIT: begin
          if (!mbox_valid) perf_stall <= perf_stall + 1'b1;
        end

        C_STREAM: begin
          wgt_rd_en     <= 1'b1;
          act_rd_en     <= 1'b1;
          wgt_rd_addr   <= c_cnt[KAW-1:0];
          act_rd_addr   <= c_cnt[KAW-1:0];
          active_m_mask <= mask_of(c_tile.m_len);
          active_n_mask <= mask_of(c_tile.n_len);
          perf_macs     <= perf_macs + 32'(c_tile.m_len * c_tile.n_len);
          c_cnt         <= c_cnt + 1'b1;
          if (c_last_read) begin
            if (c_tile.last_k)   cstate <= C_WAVE;
            else if (!mbox_valid) cstate <= C_WAIT;
          end
        end

        // Capture wave for the output block just streamed
        C_WAVE: begin
          if (wave_ok) begin
            clr_acc     <= 1'b1;
            cap_pending <= 1'b1;
            cap_timer   <= 8'(CAP_DELAY);
            cap_c_addr  <= c_tile.c_addr;
            cap_n_len   <= c_tile.n_len;
            cap_slot    <= c_tile.bias_slot;
            cap_rel     <= c_tile.rel_slot;
            if (c_tile.last_tile)  cstate <= C_FINISH;
            else if (!mbox_valid) cstate <= C_WAIT;
          end else begin
            perf_stall <= perf_stall + 1'b1;
          end
        end

        // Wait for the last output block to reach memory
        C_FINISH: begin
          if (!cap_pending && !drn_active && !drain_start && out_idle) begin
            seq_done <= 1'b1;
            cstate   <= C_IDLE;
          end
        end

        default: cstate <= C_IDLE;
      endcase

      if (c_pop) begin
        mbox_valid   <= 1'b0;
        act_buf_swap <= 1'b1;
        if (mbox.new_wgt) begin
          wgt_buf_swap <= 1'b1;
          wgt_full     <= 1'b0;
        end
        c_tile <= mbox;
        c_cnt  <= '0;
        cstate <= C_STREAM;
      end

      // ======================================================================
      // Drain scheduling
      // ======================================================================
      if (cap_pending && cap_timer > 8'd1) cap_timer <= cap_timer - 1'b1;

      if (cap_ready && drain_ready) begin
        drain_start     <= 1'b1;
        drain_c_addr    <= cap_c_addr;
        drain_n_len     <= cap_n_len;
        drain_bias_slot <= cap_slot;
        cap_pending     <= 1'b0;
        drn_active      <= 1'b1;
        drn_slot        <= cap_slot;
        drn_rel         <= cap_rel;
      end

      // A panel's bias slot is free once its last block has been drained
      // (the epilogue reads bias only while the drain is outputting columns)
      if (drain_done && drn_active) begin
        drn_active <= 1'b0;
        if (drn_rel) bias_busy[drn_slot] <= 1'b0;
      end
    end
  end

  assign seq_busy = (fstate != F_IDLE) || (cstate != C_IDLE) || seq_start;

  // new_wgt is acted on when the descriptor is popped (from mbox)
  logic _unused;
  assign _unused = &{1'b0, c_tile.new_wgt, 1'b0};

endmodule : npu_sequencer

`endif // NPU_SEQUENCER_SV
