// ============================================================================
// File: tb_npu_array_16x16.sv
// Description: Unit testbench for 16x16 Systolic Array with Skew and Drain units
//              - Validates exact 16x16 fill latency: 16 + 16 - 2 = 30 cycles
//              - Validates exact 16x16 drain latency: 16 cycles
//              - Validates full 16x16 GEMMs and arbitrary sub-tiles (M<=16, N<=16, K<=64)
//                including degenerate N=1 (FC vector product), M=1, K=1, and Conv1 shapes
//              - Generates VCD waveform for GTKWave waveform visualization
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module tb_npu_array_16x16;

  localparam int ARR_M = 16;
  localparam int ARR_N = 16;
  localparam int ACC_W = 32;
  localparam int DATA_W = 8;
  localparam int FILL_LATENCY = ARR_M + ARR_N - 2; // 30 cycles
  localparam string VECTOR_FILE = "tb/vectors/gemm_16x16_vectors.txt";

  // Clock and control
  logic clk;
  logic rst_n;
  logic en;
  logic clr_in;

  // Skew input/output interfaces
  logic signed [ARR_M*DATA_W-1:0]            a_raw;
  logic signed [ARR_N*DATA_W-1:0]            b_raw;
  wire  signed [ARR_M*DATA_W-1:0]            a_skewed;
  wire  signed [ARR_N*DATA_W-1:0]            b_skewed;
  wire         [ARR_M*ARR_N-1:0]             clr_skewed;

  // Array boundary outputs & accumulators
  wire  signed [ARR_M*DATA_W-1:0]            a_out;
  wire  signed [ARR_N*DATA_W-1:0]            b_out;
  wire  signed [ARR_M*ARR_N*ACC_W-1:0]       acc;

  // Drain interface
  logic                                      drain_start;
  wire                                       drain_busy;
  wire                                       drain_valid;
  wire  [3:0]                                drain_col_idx;
  wire  signed [ARR_M*ACC_W-1:0]             drain_data;
  wire                                       drain_done;

  // --------------------------------------------------------------------------
  // DUT Submodules
  // --------------------------------------------------------------------------

  // 1. 16x16 Skew registers
  npu_skew #(
    .ARR_M (ARR_M),
    .ARR_N (ARR_N),
    .DATA_W(DATA_W)
  ) u_skew (
    .clk        (clk),
    .rst_n      (rst_n),
    .en         (en),
    .clr_in     (clr_in),
    .a_in       (a_raw),
    .b_in       (b_raw),
    .a_skewed   (a_skewed),
    .b_skewed   (b_skewed),
    .clr_skewed (clr_skewed)
  );

  // 2. 16x16 Systolic PE Array (256 PEs)
  npu_array #(
    .ARR_M(ARR_M),
    .ARR_N(ARR_N),
    .ACC_W(ACC_W)
  ) u_array (
    .clk   (clk),
    .rst_n (rst_n),
    .en    (en),
    .clr   (clr_skewed),
    .a_in  (a_skewed),
    .b_in  (b_skewed),
    .a_out (a_out),
    .b_out (b_out),
    .acc   (acc)
  );

  // 3. 16x16 Drain Unit
  npu_drain #(
    .ARR_M(ARR_M),
    .ARR_N(ARR_N),
    .ACC_W(ACC_W)
  ) u_drain (
    .clk           (clk),
    .rst_n         (rst_n),
    .drain_start   (drain_start),
    .acc           (acc),
    .drain_busy    (drain_busy),
    .drain_valid   (drain_valid),
    .drain_col_idx (drain_col_idx),
    .drain_data    (drain_data),
    .drain_done    (drain_done)
  );

  // 100 MHz clock (10 ns period)
  initial clk = 0;
  always #5 clk = ~clk;

  // Bookkeeping
  int total_gemm_tests = 0;
  int failed_gemm_tests = 0;

  // Golden matrix storage (up to K=64)
  logic signed [DATA_W-1:0] A_mat [ARR_M][64];
  logic signed [DATA_W-1:0] B_mat [64][ARR_N];
  logic signed [ACC_W-1:0]  C_exp [ARR_M][ARR_N];
  logic signed [ACC_W-1:0]  C_act [ARR_M][ARR_N];

  // Helper task: Clear array inputs
  task automatic clear_inputs();
    clr_in = 1'b0;
    a_raw  = '0;
    b_raw  = '0;
  endtask

  // Helper task: Run a GEMM of depth K
  task automatic run_gemm(
    input int K_val,
    input bit inject_stalls = 0
  );
    int measured_drain_cycles;

    // Stream K columns/rows into skew unit
    for (int k = 0; k < K_val; k++) begin
      @(negedge clk);
      en = 1'b1;
      clr_in = (k == 0) ? 1'b1 : 1'b0;
      for (int m = 0; m < ARR_M; m++) a_raw[m*DATA_W +: DATA_W] = A_mat[m][k];
      for (int n = 0; n < ARR_N; n++) b_raw[n*DATA_W +: DATA_W] = B_mat[k][n];

      // Optionally inject a stall bubble mid-stream
      if (inject_stalls && (k == K_val/2)) begin
        @(negedge clk);
        en = 1'b0;
        clr_in = 1'b0;
        // Hold stall for 2 cycles with noisy inputs (should be ignored by array)
        for (int s = 0; s < 2; s++) begin
          for (int m = 0; m < ARR_M; m++) a_raw[m*DATA_W +: DATA_W] = 8'sh55;
          for (int n = 0; n < ARR_N; n++) b_raw[n*DATA_W +: DATA_W] = 8'shaa;
          @(negedge clk);
        end
      end
    end

    // Flush the array: drive zeros for FILL_LATENCY (30 cycles)
    for (int f = 0; f < FILL_LATENCY; f++) begin
      @(negedge clk);
      en = 1'b1;
      clear_inputs();
    end

    // Reduction is complete for all PEs. Trigger drain.
    @(negedge clk);
    drain_start = 1'b1;
    @(negedge clk);
    drain_start = 1'b0;

    // Collect drained columns on negedge clk
    measured_drain_cycles = 0;
    while (!drain_done) begin
      if (drain_valid) begin
        for (int m = 0; m < ARR_M; m++) begin
          C_act[m][drain_col_idx] = drain_data[m*ACC_W +: ACC_W];
        end
        measured_drain_cycles++;
      end
      @(negedge clk);
    end

    @(negedge clk);
  endtask

  // Main Test Process
  initial begin
    int fd;
    string token;
    int test_id, M_act, N_act, K_in, M_arr, N_arr;
    int fill_timer;
    bit pe_15_15_activated;

    $dumpfile("sim_output/tb_npu_array_16x16.vcd");
    $dumpvars(0, tb_npu_array_16x16);

    rst_n = 1'b0;
    en = 1'b1;
    drain_start = 1'b0;
    clear_inputs();

    $display("==================================================================");
    $display("  Starting 16x16 Systolic Array Unit Testbench (Milestone 3)");
    $display("==================================================================");

    // Apply reset
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) @(negedge clk);

    // ------------------------------------------------------------------------
    // Part 1: Hand-Calculated Latency Verification (16x16)
    // Hand calculation:
    //   - Fill latency = ARR_M + ARR_N - 2 = 16 + 16 - 2 = 30 cycles
    //   - Drain latency = ARR_N = 16 cycles
    // ------------------------------------------------------------------------
    $display("[CHECK] Testing Fill Latency: hand calc says PE(15,15) activates at cycle 30...");

    // Send impulse at cycle 0
    @(negedge clk);
    clr_in = 1'b1;
    for (int m = 0; m < ARR_M; m++) a_raw[m*DATA_W +: DATA_W] = 8'sd1;
    for (int n = 0; n < ARR_N; n++) b_raw[n*DATA_W +: DATA_W] = 8'sd1;

    fill_timer = 0;
    pe_15_15_activated = 0;

    // Send 0s and count cycles until PE(15,15)'s accumulator updates
    for (int c = 1; c <= 40; c++) begin
      @(negedge clk);
      clear_inputs();
      @(posedge clk);
      #1;
      if (!pe_15_15_activated && acc[(15*ARR_N + 15)*ACC_W +: ACC_W] !== '0) begin
        fill_timer = c;
        pe_15_15_activated = 1;
      end
    end

    $display("[RESULT] 16x16 Fill latency measured: %0d cycles (Expected: %0d)", fill_timer, FILL_LATENCY);
    if (fill_timer === FILL_LATENCY) begin
      $display("[PASS] Hand-calculated 16x16 fill latency matches EXACTLY (30 cycles).");
    end else begin
      $error("[FAIL] 16x16 Fill latency mismatch: got %0d, expected %0d", fill_timer, FILL_LATENCY);
      failed_gemm_tests++;
    end

    // Test 16x16 Drain Latency
    $display("[CHECK] Testing Drain Latency: hand calc says drain takes exactly ARR_N = 16 cycles...");
    @(negedge clk);
    drain_start = 1'b1;
    fill_timer = 0;
    @(negedge clk);
    drain_start = 1'b0;

    while (!drain_done && fill_timer < 30) begin
      if (drain_valid) fill_timer++;
      @(negedge clk);
    end

    $display("[RESULT] 16x16 Drain latency measured: %0d cycles (Expected: %0d)", fill_timer, ARR_N);
    if (fill_timer === ARR_N) begin
      $display("[PASS] Hand-calculated 16x16 drain latency matches EXACTLY (16 cycles).");
    end else begin
      $error("[FAIL] 16x16 Drain latency mismatch: got %0d, expected %0d", fill_timer, ARR_N);
      failed_gemm_tests++;
    end

    // ------------------------------------------------------------------------
    // Part 2: Vector-driven 16x16 GEMM Verification vs NumPy
    // ------------------------------------------------------------------------
    $display("\n[INFO] Loading 16x16 GEMM test suite from %s...", VECTOR_FILE);
    fd = $fopen(VECTOR_FILE, "r");
    if (fd == 0) begin
      $error("[ERROR] Could not open vector file: %s", VECTOR_FILE);
      $finish(1);
    end

    while (!$feof(fd)) begin
      int match;
      match = $fscanf(fd, "%s", token);
      if (match != 1) break;

      if (token == "#") begin
        string dummy;
        void'($fgets(dummy, fd));
        continue;
      end else if (token == "TEST") begin
        bit test_failed;
        test_failed = 0;
        match = $fscanf(fd, "%d %d %d %d %d %d\n", test_id, M_act, N_act, K_in, M_arr, N_arr);

        // Read A
        match = $fscanf(fd, "%s", token);
        for (int m = 0; m < ARR_M; m++) begin
          for (int k = 0; k < K_in; k++) begin
            logic [7:0] val;
            match = $fscanf(fd, "%x", val);
            A_mat[m][k] = $signed(val);
          end
        end

        // Read B
        match = $fscanf(fd, "%s", token);
        for (int k = 0; k < K_in; k++) begin
          for (int n = 0; n < ARR_N; n++) begin
            logic [7:0] val;
            match = $fscanf(fd, "%x", val);
            B_mat[k][n] = $signed(val);
          end
        end

        // Read C
        match = $fscanf(fd, "%s", token);
        for (int m = 0; m < ARR_M; m++) begin
          for (int n = 0; n < ARR_N; n++) begin
            logic [31:0] val;
            match = $fscanf(fd, "%x", val);
            C_exp[m][n] = $signed(val);
          end
        end

        total_gemm_tests++;
        run_gemm(K_in, (test_id % 4 == 0)); // inject stalls on every 4th test

        // Compare actual vs expected for all ARR_M x ARR_N elements
        for (int m = 0; m < ARR_M; m++) begin
          for (int n = 0; n < ARR_N; n++) begin
            if (C_act[m][n] !== C_exp[m][n]) begin
              $error("[MISMATCH] Test %0d (M=%0d, N=%0d, K=%0d) at C[%0d][%0d]: got %0d (0x%h), exp %0d (0x%h)",
                     test_id, M_act, N_act, K_in, m, n, C_act[m][n], C_act[m][n], C_exp[m][n], C_exp[m][n]);
              test_failed = 1;
            end
          end
        end

        if (test_failed) begin
          failed_gemm_tests++;
          $display("Test %0d FAILED", test_id);
        end else begin
          $display("[PASS] Test %0d (Active: M=%0d, N=%0d, K=%0d) bit-exact match vs NumPy.",
                   test_id, M_act, N_act, K_in);
        end
      end
    end

    $fclose(fd);

    // ------------------------------------------------------------------------
    // Part 3: Summary Report
    // ------------------------------------------------------------------------
    $display("\n==================================================================");
    $display("  Milestone 3: 16x16 Systolic Array Test Summary");
    $display("  Total GEMM tests evaluated : %0d", total_gemm_tests);
    $display("  Failed tests               : %0d", failed_gemm_tests);
    if (failed_gemm_tests == 0 && total_gemm_tests > 0) begin
      $display("  RESULT: *** ALL 16x16 TESTS PASSED *** (Bit-exact match to NumPy)");
      $display("  Fill latency:  30 cycles verified");
      $display("  Drain latency: 16 cycles verified");
      $display("==================================================================");
      $finish(0);
    end else begin
      $display("  RESULT: *** TEST FAILED ***");
      $display("==================================================================");
      $finish(1);
    end
  end

endmodule : tb_npu_array_16x16
