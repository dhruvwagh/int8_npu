// ============================================================================
// File: tb_npu_sequencer.sv
// Description: Unit Testbench for Milestone 5
//              - Multi-loop tiling engine and hardware sequencer (npu_sequencer.sv)
//              - Tests:
//                1. Standard full-tile GEMM (M=16, N=16, K=16)
//                2. Degenerate vector product / FC layer (M=10, N=1, K=784)
//                   Verifies long K reduction splitting into 7 sub-tiles with persistent accumulators
//                3. Spatial sweep with weight residency (M=16, N=196, K=64) (Conv2 shape)
//                   Verifies 13x weight traffic reduction
//                4. Partial output channels (M=8, N=32, K=25) (Conv1 shape)
//                5. Non-aligned 3D edge tiles (M=17, N=33, K=135)
//                6. Abort and error handling
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

import npu_pkg::*;

module tb_npu_sequencer;

  localparam int ARR_M      = 16;
  localparam int ARR_N      = 16;
  localparam int K_TILE_MAX = 128;
  localparam int ADDR_W     = 32;
  localparam int DATA_W     = 8;

  logic clk;
  logic rst_n;

  // CSR Control & Configuration
  logic              seq_start;
  logic              seq_abort;
  logic [15:0]       dim_m;
  logic [15:0]       dim_n;
  logic [15:0]       dim_k;
  logic [ADDR_W-1:0] a_base;
  logic [ADDR_W-1:0] b_base;
  logic [ADDR_W-1:0] c_base;
  logic [ADDR_W-1:0] bias_base;

  // Status & Performance
  logic              seq_busy;
  logic              seq_done;
  logic              seq_err;
  logic [31:0]       perf_cycles;
  logic [31:0]       perf_macs;
  logic [31:0]       perf_stall;

  // Weight Fetch Interface
  logic              fetch_wgt_start;
  logic [ADDR_W-1:0] fetch_wgt_base_addr;
  logic [7:0]        fetch_wgt_words;
  logic              fetch_wgt_busy;
  logic              fetch_wgt_done;

  // Activation Fetch Interface
  logic              fetch_act_start;
  logic [ADDR_W-1:0] fetch_act_base_addr;
  logic [7:0]        fetch_act_words;
  logic              fetch_act_busy;
  logic              fetch_act_done;

  // Bias Fetch Interface
  logic              fetch_bias_start;
  logic [ADDR_W-1:0] fetch_bias_base_addr;
  logic              fetch_bias_busy;
  logic              fetch_bias_done;

  // Buffer Swap & Bank Tracking
  logic              wgt_buf_swap;
  logic              act_buf_swap;
  logic              wgt_buf_cur_bank;
  logic              act_buf_cur_bank;

  // Buffer Compute Read Interface
  logic              wgt_rd_en;
  logic [$clog2(K_TILE_MAX)-1:0] wgt_rd_addr;
  logic              act_rd_en;
  logic [$clog2(K_TILE_MAX)-1:0] act_rd_addr;

  // Compute Control
  logic              array_en;
  logic              clr_acc;
  logic              drain_start;
  logic              drain_busy;
  logic              drain_done;
  logic              drain_ready;
  logic [ADDR_W-1:0] drain_c_addr;
  logic [7:0]        drain_n_len;

  // Tile Descriptor & Masks
  logic              tile_valid;
  tile_desc_t        tile_desc;
  logic [ARR_M-1:0]  active_m_mask;
  logic [ARR_N-1:0]  active_n_mask;

  // --------------------------------------------------------------------------
  // DUT Instantiation
  // --------------------------------------------------------------------------
  npu_sequencer #(
    .ARR_M      (ARR_M),
    .ARR_N      (ARR_N),
    .K_TILE_MAX (K_TILE_MAX),
    .ADDR_W     (ADDR_W),
    .DATA_W     (DATA_W)
  ) dut (
    .clk                 (clk),
    .rst_n               (rst_n),
    .seq_start           (seq_start),
    .seq_abort           (seq_abort),
    .dim_m               (dim_m),
    .dim_n               (dim_n),
    .dim_k               (dim_k),
    .a_base              (a_base),
    .b_base              (b_base),
    .c_base              (c_base),
    .bias_base           (bias_base),
    .seq_busy            (seq_busy),
    .seq_done            (seq_done),
    .seq_err             (seq_err),
    .perf_cycles         (perf_cycles),
    .perf_macs           (perf_macs),
    .perf_stall          (perf_stall),
    .perf_clr            (1'b0),
    .fetch_wgt_start     (fetch_wgt_start),
    .fetch_wgt_base_addr (fetch_wgt_base_addr),
    .fetch_wgt_words     (fetch_wgt_words),
    .fetch_wgt_busy      (fetch_wgt_busy),
    .fetch_wgt_done      (fetch_wgt_done),
    .fetch_act_start     (fetch_act_start),
    .fetch_act_base_addr (fetch_act_base_addr),
    .fetch_act_words     (fetch_act_words),
    .fetch_act_busy      (fetch_act_busy),
    .fetch_act_done      (fetch_act_done),
    .fetch_bias_start    (fetch_bias_start),
    .fetch_bias_base_addr(fetch_bias_base_addr),
    .fetch_bias_busy     (fetch_bias_busy),
    .fetch_bias_done     (fetch_bias_done),
    .wgt_buf_swap        (wgt_buf_swap),
    .act_buf_swap        (act_buf_swap),
    .wgt_buf_cur_bank    (wgt_buf_cur_bank),
    .act_buf_cur_bank    (act_buf_cur_bank),
    .wgt_rd_en           (wgt_rd_en),
    .wgt_rd_addr         (wgt_rd_addr),
    .act_rd_en           (act_rd_en),
    .act_rd_addr         (act_rd_addr),
    .array_en            (array_en),
    .clr_acc             (clr_acc),
    .drain_start         (drain_start),
    .drain_done          (drain_done),
    .drain_ready         (drain_ready),
    .drain_c_addr        (drain_c_addr),
    .drain_n_len         (drain_n_len),
    .tile_valid          (tile_valid),
    .tile_desc           (tile_desc),
    .active_m_mask       (active_m_mask),
    .active_n_mask       (active_n_mask)
  );

  // --------------------------------------------------------------------------
  // Clock Generator (100 MHz, 10ns period)
  // --------------------------------------------------------------------------
  initial clk = 0;
  always #5 clk = ~clk;

  // --------------------------------------------------------------------------
  // Emulated Subsystems (Fetch Engines, Buffers, Drain)
  // --------------------------------------------------------------------------

  // Weight fetch emulator (completes in 2 cycles)
  int wgt_fetch_timer;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fetch_wgt_busy  <= 1'b0;
      fetch_wgt_done  <= 1'b0;
      wgt_fetch_timer <= 0;
    end else begin
      fetch_wgt_done <= 1'b0;
      if (fetch_wgt_start) begin
        fetch_wgt_busy  <= 1'b1;
        wgt_fetch_timer <= 2;
      end else if (fetch_wgt_busy) begin
        if (wgt_fetch_timer > 1) begin
          wgt_fetch_timer <= wgt_fetch_timer - 1;
        end else begin
          fetch_wgt_busy <= 1'b0;
          fetch_wgt_done <= 1'b1;
        end
      end
    end
  end

  // Activation fetch emulator (completes in 2 cycles)
  int act_fetch_timer;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fetch_act_busy  <= 1'b0;
      fetch_act_done  <= 1'b0;
      act_fetch_timer <= 0;
    end else begin
      fetch_act_done <= 1'b0;
      if (fetch_act_start) begin
        fetch_act_busy  <= 1'b1;
        act_fetch_timer <= 2;
      end else if (fetch_act_busy) begin
        if (act_fetch_timer > 1) begin
          act_fetch_timer <= act_fetch_timer - 1;
        end else begin
          fetch_act_busy <= 1'b0;
          fetch_act_done <= 1'b1;
        end
      end
    end
  end

  // Bias fetch emulator (completes in 2 cycles)
  int bias_fetch_timer;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fetch_bias_busy  <= 1'b0;
      fetch_bias_done  <= 1'b0;
      bias_fetch_timer <= 0;
    end else begin
      fetch_bias_done <= 1'b0;
      if (fetch_bias_start) begin
        fetch_bias_busy  <= 1'b1;
        bias_fetch_timer <= 2;
      end else if (fetch_bias_busy) begin
        if (bias_fetch_timer > 1) begin
          bias_fetch_timer <= bias_fetch_timer - 1;
        end else begin
          fetch_bias_busy <= 1'b0;
          fetch_bias_done <= 1'b1;
        end
      end
    end
  end

  // Output path is free again once the drained tile has been "written back"
  // (emulated as drain + ARR_N writeback cycles)
  int wb_timer;
  assign drain_ready = !drain_busy && (wb_timer == 0);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wb_timer <= 0;
    end else if (drain_done) begin
      wb_timer <= ARR_N;
    end else if (wb_timer > 0) begin
      wb_timer <= wb_timer - 1;
    end
  end

  // Buffer bank flip-flops
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wgt_buf_cur_bank <= 1'b0;
      act_buf_cur_bank <= 1'b0;
    end else begin
      if (wgt_buf_swap) wgt_buf_cur_bank <= ~wgt_buf_cur_bank;
      if (act_buf_swap) act_buf_cur_bank <= ~act_buf_cur_bank;
    end
  end

  // Drain emulator (completes in ARR_N = 16 cycles)
  int drain_timer;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      drain_busy  <= 1'b0;
      drain_done  <= 1'b0;
      drain_timer <= 0;
    end else begin
      drain_done <= 1'b0;
      if (drain_start) begin
        drain_busy  <= 1'b1;
        drain_timer <= ARR_N;
      end else if (drain_busy) begin
        if (drain_timer > 1) begin
          drain_timer <= drain_timer - 1;
        end else begin
          drain_busy <= 1'b0;
          drain_done <= 1'b1;
        end
      end
    end
  end

  // --------------------------------------------------------------------------
  // Verification Diagnostics
  // --------------------------------------------------------------------------
  int total_errors = 0;
  int test_count = 0;

  // Trackers for current test run
  int tiles_launched;
  int wgt_fetches_issued;
  int act_fetches_issued;
  int clr_acc_count;
  int drain_start_count;
  int bias_fetches_issued;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tiles_launched     <= 0;
      wgt_fetches_issued <= 0;
      act_fetches_issued <= 0;
      clr_acc_count      <= 0;
      drain_start_count  <= 0;
      bias_fetches_issued <= 0;
    end else begin
      if (tile_valid)      tiles_launched     <= tiles_launched + 1;
      if (fetch_wgt_start) wgt_fetches_issued <= wgt_fetches_issued + 1;
      if (fetch_act_start) act_fetches_issued <= act_fetches_issued + 1;
      if (clr_acc)         clr_acc_count      <= clr_acc_count + 1;
      if (drain_start)     drain_start_count  <= drain_start_count + 1;
      if (fetch_bias_start) bias_fetches_issued <= bias_fetches_issued + 1;
    end
  end

  // Helper task: launch layer and wait for completion
  task run_layer(
    input string          name,
    input logic [15:0]    m,
    input logic [15:0]    n,
    input logic [15:0]    k,
    input logic [31:0]    expected_macs,
    input int             expected_tiles,
    input int             expected_wgt_fetches,
    input int             expected_drains
  );
    test_count++;
    $display("\n[TEST %0d] %s: M=%0d, N=%0d, K=%0d", test_count, name, m, n, k);

    tiles_launched     = 0;
    wgt_fetches_issued = 0;
    act_fetches_issued = 0;
    clr_acc_count      = 0;
    drain_start_count  = 0;
    bias_fetches_issued = 0;

    @(posedge clk);
    dim_m     <= m;
    dim_n     <= n;
    dim_k     <= k;
    a_base    <= 32'h0001_0000;
    b_base    <= 32'h0002_0000;
    c_base    <= 32'h0003_0000;
    bias_base <= 32'h0004_0000;
    seq_start <= 1'b1;

    @(posedge clk);
    seq_start <= 1'b0;

    // Wait for seq_done pulse with timeout protection
    fork
      begin
        while (!seq_done) @(posedge clk);
      end
      begin
        repeat (50000) @(posedge clk);
        $display("[FAIL] Timeout waiting for seq_done in %s at time %0t", name, $time);
        total_errors++;
      end
    join_any
    disable fork;

    #1;
    $display("  -> Layer finished: perf_cycles=%0d, perf_macs=%0d, perf_stall=%0d",
             perf_cycles, perf_macs, perf_stall);
    $display("  -> Tiles launched=%0d, Wgt fetches=%0d, Act fetches=%0d, Drains=%0d",
             tiles_launched, wgt_fetches_issued, act_fetches_issued, drain_start_count);

    // Audit performance MAC count
    if (perf_macs !== expected_macs) begin
      $display("[FAIL] Expected perf_macs=%0d, got %0d", expected_macs, perf_macs);
      total_errors++;
    end

    // Audit tile count
    if (tiles_launched !== expected_tiles) begin
      $display("[FAIL] Expected tiles_launched=%0d, got %0d", expected_tiles, tiles_launched);
      total_errors++;
    end

    // Audit weight fetch count (verifying residency)
    if (wgt_fetches_issued !== expected_wgt_fetches) begin
      $display("[FAIL] Expected wgt_fetches=%0d, got %0d", expected_wgt_fetches, wgt_fetches_issued);
      total_errors++;
    end

    // Audit bias fetch count: exactly one per M panel
    if (bias_fetches_issued !== (m + ARR_M - 1) / ARR_M) begin
      $display("[FAIL] Expected bias fetches=%0d, got %0d", (m + ARR_M - 1) / ARR_M, bias_fetches_issued);
      total_errors++;
    end

    // Audit drain count
    if (drain_start_count !== expected_drains) begin
      $display("[FAIL] Expected drain_count=%0d, got %0d", expected_drains, drain_start_count);
      total_errors++;
    end

    if (total_errors == 0) begin
      $display("[PASS] %s completed successfully!", name);
    end
  endtask

  // --------------------------------------------------------------------------
  // Main Test Sequence
  // --------------------------------------------------------------------------
  initial begin
    $dumpfile("sim_output/tb_npu_sequencer.vcd");
    $dumpvars(0, tb_npu_sequencer);

    $display("==================================================================");
    $display("  Milestone 5: Multi-Loop Tiling Sequencer Verification");
    $display("==================================================================");

    rst_n     = 1'b0;
    seq_start = 1'b0;
    seq_abort = 1'b0;
    dim_m     = '0;
    dim_n     = '0;
    dim_k     = '0;
    a_base    = '0;
    b_base    = '0;
    c_base    = '0;
    bias_base = '0;

    #20;
    @(negedge clk);
    rst_n = 1'b1;
    #20;

    // ------------------------------------------------------------------------
    // Test 1: Standard Full-Tile GEMM (M=16, N=16, K=16)
    // ------------------------------------------------------------------------
    // Total MACs: 16 * 16 * 16 = 4096
    // Tiles: 1 tile, 1 weight fetch, 1 drain
    run_layer("Standard Full-Tile GEMM", 16, 16, 16, 4096, 1, 1, 1);

    // ------------------------------------------------------------------------
    // Test 2: Degenerate Vector Product / FC Layer (M=10, N=1, K=784)
    // ------------------------------------------------------------------------
    // M=10 (partial rows), N=1 (degenerate column), K=784 (K > K_TILE_MAX=128)
    // K splits into 7 sub-tiles: 6 of 128 + 1 of 16
    // Total MACs: 10 * 1 * 784 = 7840
    // Tiles: 7 tiles
    // Weight fetches: 7 (one per K slice)
    // Drains: 1 (ONLY at the very end when K reduction finishes!)
    run_layer("FC Layer (K=784 Reduction Splitting)", 10, 1, 784, 7840, 7, 7, 1);

    // ------------------------------------------------------------------------
    // Test 3: Conv2 Spatial Sweep with Weight Residency (M=16, N=196, K=64)
    // ------------------------------------------------------------------------
    // M=16, N=196 (13 tiles: 12 full + 1 partial of 4), K=64 <= 128
    // Total MACs: 16 * 196 * 64 = 200,704
    // Tiles: 13 tiles
    // Weight fetches: EXACTLY 1! (13x traffic reduction achieved!)
    // Drains: 13 (one per spatial output tile)
    run_layer("Conv2 Spatial Sweep with Weight Residency", 16, 196, 64, 200704, 13, 1, 13);

    // ------------------------------------------------------------------------
    // Test 4: Conv1 Partial Channels (M=8, N=32, K=25)
    // ------------------------------------------------------------------------
    // M=8 (partial rows), N=32 (2 tiles of 16), K=25
    // Total MACs: 8 * 32 * 25 = 6400
    // Tiles: 2 tiles
    // Weight fetches: EXACTLY 1! (weight stays resident across the 2 N tiles)
    // Drains: 2
    run_layer("Conv1 Partial Channels", 8, 32, 25, 6400, 2, 1, 2);

    // ------------------------------------------------------------------------
    // Test 5: Non-Aligned 3D Partial Edge Tiles (M=17, N=33, K=135)
    // ------------------------------------------------------------------------
    // M=17 (2 tiles: 16, 1), N=33 (3 tiles: 16, 16, 1), K=135 (2 slices: 128, 7)
    // Total MACs: 17 * 33 * 135 = 75,735
    // Both K > 128 and N > 16: auto-selects K inner (M -> N -> K)
    // Total tiles: 2 * 3 * 2 = 12 tiles
    // Drains: 2 * 3 = 6 (one per (m0, n0) after its 2 K slices complete)
    // Weight fetches: 12
    run_layer("Non-Aligned 3D Edge Tiles", 17, 33, 135, 75735, 12, 12, 6);

    // ------------------------------------------------------------------------
    // Test 6: Abort & Error Handling
    // ------------------------------------------------------------------------
    test_count++;
    $display("\n[TEST %0d] Error Handling (M=0 Invalid Dimension)", test_count);
    begin
      @(posedge clk);
      dim_m     <= 16'd0; // Invalid!
      dim_n     <= 16'd16;
      dim_k     <= 16'd16;
      seq_start <= 1'b1;
      @(posedge clk);
      seq_start <= 1'b0;
      #1;
      if (!seq_err) begin
        $display("[FAIL] Expected seq_err=1 for dim_m=0, got %0d", seq_err);
        total_errors++;
      end
      if (seq_busy) begin
        $display("[FAIL] Expected seq_busy=0 on error, got %0d", seq_busy);
        total_errors++;
      end
      $display("[PASS] Error handling verified: seq_err correctly asserted.");
    end

    test_count++;
    $display("\n[TEST %0d] Abort Mid-Execution", test_count);
    begin
      @(posedge clk);
      dim_m     <= 16'd16;
      dim_n     <= 16'd196;
      dim_k     <= 16'd64;
      seq_start <= 1'b1;
      @(posedge clk);
      seq_start <= 1'b0;

      // Let it run for 100 cycles then abort
      repeat (100) @(posedge clk);
      if (!seq_busy) begin
        $display("[FAIL] Expected seq_busy=1 before abort, got %0d", seq_busy);
        total_errors++;
      end

      @(posedge clk);
      seq_abort <= 1'b1;
      @(posedge clk);
      seq_abort <= 1'b0;

      @(posedge clk);
      #1;
      if (seq_busy) begin
        $display("[FAIL] Expected seq_busy=0 after abort, got %0d", seq_busy);
        total_errors++;
      end
      $display("[PASS] Abort handling verified: sequencer cleanly returned to IDLE.");
    end

    // ------------------------------------------------------------------------
    // Final Summary
    // ------------------------------------------------------------------------
    $display("\n==================================================================");
    if (total_errors == 0) begin
      $display("  ALL %0d TESTS PASSED! (0 errors)", test_count);
      $display("  Milestone 5 Exit Criteria MET:");
      $display("  1. Arbitrary M/N/K tiling verified including partial edge tiles.");
      $display("  2. K > K_TILE reduction splitting verified with accumulators held live.");
      $display("  3. Weight tile residency verified with 13x traffic reduction on Conv2.");
      $display("  4. Performance MAC counters match exact mathematical product (M * N * K).");
      $display("  5. Error reporting and abort mechanisms operate cleanly.");
      $display("==================================================================");
    end else begin
      $display("  TEST FAILED WITH %0d ERRORS!", total_errors);
      $display("==================================================================");
      $fatal(1, "Milestone 5 testbench encountered %0d errors", total_errors);
    end

    #50;
    $finish;
  end

endmodule
