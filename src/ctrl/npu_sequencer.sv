// ============================================================================
// File: npu_sequencer.sv
// Description: Multi-Loop Tiling Engine and Hardware Sequencer
//              - Orchestrates nested M, K, N loops in Weight-Residency order (M -> K -> N)
//              - Supports arbitrary M, N, K layer shapes including partial edge tiles
//              - Supports long reduction splitting (K > K_TILE_MAX) with persistent accumulators
//              - Manages double-buffering handshakes with npu_fetch, npu_buf, and npu_array
//              - Hardware performance profiling: total cycles, MACs, stall cycles
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
  output logic [31:0]         perf_stall,      // Total cycles stalled waiting for memory

  // Weight Fetch Engine Interface
  output logic                fetch_wgt_start,
  output logic [ADDR_W-1:0]   fetch_wgt_base_addr,
  output logic [7:0]          fetch_wgt_words,
  input  logic                fetch_wgt_busy,
  input  logic                fetch_wgt_done,

  // Activation Fetch Engine Interface
  output logic                fetch_act_start,
  output logic [ADDR_W-1:0]   fetch_act_base_addr,
  output logic [7:0]          fetch_act_words,
  input  logic                fetch_act_busy,
  input  logic                fetch_act_done,

  // Ping-Pong Buffer Control
  output logic                wgt_buf_swap,
  output logic                act_buf_swap,
  input  logic                wgt_buf_cur_bank,
  input  logic                act_buf_cur_bank,

  // Buffer Compute Read Address Interface
  output logic                wgt_rd_en,
  output logic [$clog2(K_TILE_MAX)-1:0] wgt_rd_addr,
  output logic                act_rd_en,
  output logic [$clog2(K_TILE_MAX)-1:0] act_rd_addr,

  // Compute Array Control
  output logic                array_en,        // Skew and PE clock enable
  output logic                clr_acc,         // Pulse to clear accumulators at tile start
  output logic                drain_start,     // Pulse to snapshot accumulators
  input  logic                drain_busy,      // Drain module is busy
  input  logic                drain_done,      // Drain module finished

  // Tile Descriptor & Masking Interface
  output logic                tile_valid,      // Pulse when a tile starts computing
  output npu_pkg::tile_desc_t tile_desc,       // Current tile parameters
  output logic [ARR_M-1:0]    active_m_mask,   // 1 for active rows, 0 for padded rows
  output logic [ARR_N-1:0]    active_n_mask    // 1 for active cols, 0 for padded cols
);

  localparam int FILL_LATENCY = ARR_M + ARR_N - 1; // Skew + array latency to PE(15,15)

  // --------------------------------------------------------------------------
  // Sequencer FSM States
  // --------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE,
    S_FETCH_WGT,
    S_FETCH_ACT,
    S_WAIT_FETCH,
    S_BUFFER_SWAP,
    S_COMPUTE_PRIME,
    S_COMPUTE_STREAM,
    S_COMPUTE_FILL,
    S_DRAIN_WAIT,
    S_NEXT_TILE,
    S_DONE
  } state_t;

  state_t state;

  // Latched layer configuration registers
  logic [15:0]       reg_m, reg_n, reg_k;
  logic [ADDR_W-1:0] reg_a_base, reg_b_base, reg_c_base, reg_bias_base;

  // Tile loop coordinate registers
  logic [15:0] m0, n0, k0;
  logic [7:0]  cur_m_len, cur_n_len, cur_k_len;
  logic        cur_first_k, cur_last_k;

  // Weight residency tracking: 1 if active weight bank holds current (m0, k0)
  logic        wgt_resident;

  // Fetch pending tracking
  logic        wgt_fetch_pending;
  logic        act_fetch_pending;

  // Execution counters
  logic [7:0]  comp_k_cnt;
  logic [7:0]  fill_cnt;

  // --------------------------------------------------------------------------
  // Combinational Tile Geometry & Edge Calculations
  // --------------------------------------------------------------------------
  logic [15:0] rem_m, rem_n, rem_k;
  logic [7:0]  next_m_len, next_n_len, next_k_len;
  logic        k_split_multitile;

  always_comb begin
    rem_m = (reg_m > m0) ? (reg_m - m0) : 16'd0;
    rem_n = (reg_n > n0) ? (reg_n - n0) : 16'd0;
    rem_k = (reg_k > k0) ? (reg_k - k0) : 16'd0;

    next_m_len = (rem_m < 16'(ARR_M))      ? 8'(rem_m) : 8'(ARR_M);
    next_n_len = (rem_n < 16'(ARR_N))      ? 8'(rem_n) : 8'(ARR_N);
    next_k_len = (rem_k < 16'(K_TILE_MAX)) ? 8'(rem_k) : 8'(K_TILE_MAX);

    // Auto-select loop ordering:
    // If K > K_TILE_MAX and N > ARR_N, prioritize accumulator retention by running K inner.
    // Otherwise, prioritize weight residency (M -> K -> N).
    k_split_multitile = (reg_k > 16'(K_TILE_MAX)) && (reg_n > 16'(ARR_N));
  end

  // Active element masks (for zero-padding sub-tiles)
  always_comb begin
    for (int i = 0; i < ARR_M; i++) begin
      active_m_mask[i] = (8'(i) < cur_m_len);
    end
    for (int j = 0; j < ARR_N; j++) begin
      active_n_mask[j] = (8'(j) < cur_n_len);
    end
  end

  // --------------------------------------------------------------------------
  // Main Sequencer FSM
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state               <= S_IDLE;
      seq_done            <= 1'b0;
      seq_err             <= 1'b0;
      perf_cycles         <= '0;
      perf_macs           <= '0;
      perf_stall          <= '0;
      m0                  <= '0;
      n0                  <= '0;
      k0                  <= '0;
      cur_m_len           <= '0;
      cur_n_len           <= '0;
      cur_k_len           <= '0;
      cur_first_k         <= 1'b0;
      cur_last_k          <= 1'b0;
      wgt_resident        <= 1'b0;
      wgt_fetch_pending   <= 1'b0;
      act_fetch_pending   <= 1'b0;
      comp_k_cnt          <= '0;
      fill_cnt            <= '0;
      fetch_wgt_start     <= 1'b0;
      fetch_wgt_base_addr <= '0;
      fetch_wgt_words     <= '0;
      fetch_act_start     <= 1'b0;
      fetch_act_base_addr <= '0;
      fetch_act_words     <= '0;
      wgt_buf_swap        <= 1'b0;
      act_buf_swap        <= 1'b0;
      wgt_rd_en           <= 1'b0;
      wgt_rd_addr         <= '0;
      act_rd_en           <= 1'b0;
      act_rd_addr         <= '0;
      array_en            <= 1'b0;
      clr_acc             <= 1'b0;
      drain_start         <= 1'b0;
      tile_valid          <= 1'b0;
      tile_desc           <= '0;
      reg_m               <= '0;
      reg_n               <= '0;
      reg_k               <= '0;
      reg_a_base          <= '0;
      reg_b_base          <= '0;
      reg_c_base          <= '0;
      reg_bias_base       <= '0;
    end else if (seq_abort) begin
      state               <= S_IDLE;
      seq_done            <= 1'b0;
      fetch_wgt_start     <= 1'b0;
      fetch_act_start     <= 1'b0;
      wgt_buf_swap        <= 1'b0;
      act_buf_swap        <= 1'b0;
      wgt_rd_en           <= 1'b0;
      act_rd_en           <= 1'b0;
      array_en            <= 1'b0;
      clr_acc             <= 1'b0;
      drain_start         <= 1'b0;
      tile_valid          <= 1'b0;
      wgt_fetch_pending   <= 1'b0;
      act_fetch_pending   <= 1'b0;
    end else begin
      // Default single-cycle strobes
      seq_done        <= 1'b0;
      seq_err         <= 1'b0;
      fetch_wgt_start <= 1'b0;
      fetch_act_start <= 1'b0;
      wgt_buf_swap    <= 1'b0;
      act_buf_swap    <= 1'b0;
      clr_acc         <= 1'b0;
      drain_start     <= 1'b0;
      tile_valid      <= 1'b0;

      // Track asynchronous fetch completion
      if (fetch_wgt_done) begin
        wgt_fetch_pending <= 1'b0;
      end
      if (fetch_act_done) begin
        act_fetch_pending <= 1'b0;
      end

      // Global performance cycle counter
      if (state != S_IDLE) begin
        perf_cycles <= perf_cycles + 1'b1;
      end

      case (state)
        // --------------------------------------------------------------------
        // IDLE: Wait for start command from CSR
        // --------------------------------------------------------------------
        S_IDLE: begin
          if (seq_start) begin
            if (dim_m == 0 || dim_n == 0 || dim_k == 0) begin
              seq_err <= 1'b1;
            end else begin
              // Latch layer configuration
              reg_m             <= dim_m;
              reg_n             <= dim_n;
              reg_k             <= dim_k;
              reg_a_base        <= a_base;
              reg_b_base        <= b_base;
              reg_c_base        <= c_base;
              reg_bias_base     <= bias_base;

              // Reset performance counters
              perf_cycles       <= '0;
              perf_macs         <= '0;
              perf_stall        <= '0;

              // Initialize loop indices (M -> K -> N order)
              m0                <= '0;
              n0                <= '0;
              k0                <= '0;
              wgt_resident      <= 1'b0;
              wgt_fetch_pending <= 1'b0;
              act_fetch_pending <= 1'b0;

              state             <= S_FETCH_WGT;
            end
          end
        end

        // --------------------------------------------------------------------
        // FETCH WEIGHT: Trigger weight fetch if not currently resident
        // --------------------------------------------------------------------
        S_FETCH_WGT: begin
          cur_m_len   <= next_m_len;
          cur_n_len   <= next_n_len;
          cur_k_len   <= next_k_len;
          cur_first_k <= (k0 == 0);
          cur_last_k  <= ((k0 + 16'(next_k_len)) >= reg_k);

          if (!wgt_resident) begin
            fetch_wgt_start     <= 1'b1;
            wgt_fetch_pending   <= 1'b1;
            // Byte address in tensor memory
            fetch_wgt_base_addr <= reg_a_base + ((ADDR_W'(m0) * ADDR_W'(reg_k) + ADDR_W'(k0)) << 0);
            fetch_wgt_words     <= next_k_len;
          end
          state <= S_FETCH_ACT;
        end

        // --------------------------------------------------------------------
        // FETCH ACTIVATION: Trigger activation tile fetch into idle bank
        // --------------------------------------------------------------------
        S_FETCH_ACT: begin
          cur_m_len           <= next_m_len;
          cur_n_len           <= next_n_len;
          cur_k_len           <= next_k_len;
          cur_first_k         <= (k0 == 0);
          cur_last_k          <= ((k0 + 16'(next_k_len)) >= reg_k);

          fetch_act_start     <= 1'b1;
          act_fetch_pending   <= 1'b1;
          fetch_act_base_addr <= reg_b_base + ((ADDR_W'(k0) * ADDR_W'(reg_n) + ADDR_W'(n0)) << 0);
          fetch_act_words     <= next_k_len;
          state               <= S_WAIT_FETCH;
        end

        // --------------------------------------------------------------------
        // WAIT FETCH: Synchronize background DMA fetch with buffer readiness
        // --------------------------------------------------------------------
        S_WAIT_FETCH: begin
          if (act_fetch_pending || fetch_act_busy ||
              (!wgt_resident && (wgt_fetch_pending || fetch_wgt_busy))) begin
            perf_stall <= perf_stall + 1'b1;
          end else begin
            state <= S_BUFFER_SWAP;
          end
        end

        // --------------------------------------------------------------------
        // BUFFER SWAP: Atomic bank swap to expose newly fetched data to compute
        // --------------------------------------------------------------------
        S_BUFFER_SWAP: begin
          act_buf_swap <= 1'b1;
          if (!wgt_resident) begin
            wgt_buf_swap <= 1'b1;
            wgt_resident <= 1'b1; // Weight tile is now resident in active bank
          end
          state <= S_COMPUTE_PRIME;
        end

        // --------------------------------------------------------------------
        // COMPUTE PRIME: Issue tile descriptor and prime buffer read address 0
        // --------------------------------------------------------------------
        S_COMPUTE_PRIME: begin
          tile_valid          <= 1'b1;
          tile_desc.m0        <= m0;
          tile_desc.n0        <= n0;
          tile_desc.k0        <= k0;
          tile_desc.m_len     <= cur_m_len;
          tile_desc.n_len     <= cur_n_len;
          tile_desc.k_len     <= cur_k_len;
          tile_desc.first_k   <= cur_first_k;
          tile_desc.last_k    <= cur_last_k;
          tile_desc.a_base    <= reg_a_base;
          tile_desc.b_base    <= reg_b_base;
          tile_desc.c_base    <= reg_c_base;
          tile_desc.bias_base <= reg_bias_base;

          // Clear accumulators on first K tile of this (m0, n0)
          clr_acc    <= cur_first_k;
          array_en   <= 1'b1;

          // Prime address 0
          wgt_rd_en   <= 1'b1;
          wgt_rd_addr <= '0;
          act_rd_en   <= 1'b1;
          act_rd_addr <= '0;
          comp_k_cnt  <= 8'd1;

          state <= S_COMPUTE_STREAM;
        end

        // --------------------------------------------------------------------
        // COMPUTE STREAM: Stream k_len words from buffers into array at 1 beat/cycle
        // --------------------------------------------------------------------
        S_COMPUTE_STREAM: begin
          array_en <= 1'b1;
          // Track MAC operations issued (m_len * n_len per k cycle)
          perf_macs <= perf_macs + 32'(cur_m_len * cur_n_len);

          if (comp_k_cnt < cur_k_len) begin
            wgt_rd_en   <= 1'b1;
            wgt_rd_addr <= comp_k_cnt[$clog2(K_TILE_MAX)-1:0];
            act_rd_en   <= 1'b1;
            act_rd_addr <= comp_k_cnt[$clog2(K_TILE_MAX)-1:0];
            comp_k_cnt  <= comp_k_cnt + 1'b1;
          end else begin
            wgt_rd_en  <= 1'b0;
            act_rd_en  <= 1'b0;
            fill_cnt   <= '0;
            state      <= S_COMPUTE_FILL;
          end
        end

        // --------------------------------------------------------------------
        // COMPUTE FILL: Wait for systolic pipeline delay (31 cycles for 16x16)
        // --------------------------------------------------------------------
        S_COMPUTE_FILL: begin
          array_en <= 1'b1;
          fill_cnt <= fill_cnt + 1'b1;

          if (fill_cnt >= 8'(FILL_LATENCY)) begin
            array_en <= 1'b0;
            if (cur_last_k) begin
              // Final reduction step: trigger accumulator readout
              drain_start <= 1'b1;
              state       <= S_DRAIN_WAIT;
            end else begin
              // Partial reduction: keep accumulators live, advance to next tile
              state <= S_NEXT_TILE;
            end
          end
        end

        // --------------------------------------------------------------------
        // DRAIN WAIT: Wait for column drain module to snapshot accumulators
        // --------------------------------------------------------------------
        S_DRAIN_WAIT: begin
          // As soon as drain snapshots (drain_busy goes high then drops or drain_done),
          // next tile can proceed.
          if (drain_done || !drain_busy) begin
            state <= S_NEXT_TILE;
          end
        end

        // --------------------------------------------------------------------
        // NEXT TILE: Multi-loop index incrementing
        // --------------------------------------------------------------------
        S_NEXT_TILE: begin
          if (k_split_multitile) begin
            // K is innermost loop: complete full K reduction for (m0, n0) before moving to next n0
            if ((k0 + 16'(K_TILE_MAX)) < reg_k) begin
              k0           <= k0 + 16'(K_TILE_MAX);
              wgt_resident <= 1'b0;
              state        <= S_FETCH_WGT;
            end else begin
              k0 <= '0;
              if ((n0 + 16'(ARR_N)) < reg_n) begin
                n0           <= n0 + 16'(ARR_N);
                wgt_resident <= 1'b0;
                state        <= S_FETCH_WGT;
              end else begin
                n0 <= '0;
                if ((m0 + 16'(ARR_M)) < reg_m) begin
                  m0           <= m0 + 16'(ARR_M);
                  wgt_resident <= 1'b0;
                  state        <= S_FETCH_WGT;
                end else begin
                  state <= S_DONE;
                end
              end
            end
          end else begin
            // Standard Weight-Residency Priority: M -> K -> N
            if ((n0 + 16'(ARR_N)) < reg_n) begin
              n0    <= n0 + 16'(ARR_N);
              state <= S_FETCH_ACT; // Weight tile is resident; fetch next activation tile
            end else begin
              n0 <= '0;
              if ((k0 + 16'(K_TILE_MAX)) < reg_k) begin
                k0           <= k0 + 16'(K_TILE_MAX);
                wgt_resident <= 1'b0; // Moving to new K slice: must fetch new weights
                state        <= S_FETCH_WGT;
              end else begin
                k0 <= '0;
                if ((m0 + 16'(ARR_M)) < reg_m) begin
                  m0           <= m0 + 16'(ARR_M);
                  wgt_resident <= 1'b0; // Moving to new M channels: must fetch new weights
                  state        <= S_FETCH_WGT;
                end else begin
                  state <= S_DONE;
                end
              end
            end
          end
        end

        // --------------------------------------------------------------------
        // DONE: Layer execution complete
        // --------------------------------------------------------------------
        S_DONE: begin
          seq_done <= 1'b1;
          state    <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  assign seq_busy = (state != S_IDLE) | seq_start;

  // Suppress unused lint warnings for bank index signals (used for debug/inspection)
  logic _unused;
  assign _unused = &{1'b0, wgt_buf_cur_bank, act_buf_cur_bank, DATA_W, 1'b0};

endmodule : npu_sequencer

`endif // NPU_SEQUENCER_SV
