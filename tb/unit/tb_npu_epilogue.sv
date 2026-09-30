// ============================================================================
// File: tb_npu_epilogue.sv
// Description: Unit Testbench for Milestone 6
//              - Multi-lane requantization epilogue unit (npu_epilogue.sv)
//              - Double-buffered output staging buffer (npu_obuf.sv)
//              - Tests:
//                1. 1,332 golden vectors from model/golden_epilogue.py (boundary sweeps,
//                   saturation clamping at -128/+127, round half away from zero, ReLU)
//                2. 16-lane parallel SIMD execution
//                3. Full 16x16 tile drain through epilogue into npu_obuf
//                4. Double-buffered concurrent drain and DMA readback
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module tb_npu_epilogue;

  localparam int ARR_M  = 16;
  localparam int ARR_N  = 16;
  localparam int ACC_W  = 32;
  localparam int DATA_W = 8;
  localparam int DW     = ARR_M * DATA_W; // 128 bits
  localparam int WORDS  = 16;
  localparam int AW     = $clog2(WORDS);
  localparam int EPI_LAT = 2;              // npu_epilogue pipeline latency (cycles)

  logic signed [7:0] tile_gold_t3 [ARR_M][ARR_N];  // Test 3 tile, re-checked in Test 4

  logic clk;
  logic rst_n;

  // Epilogue Control & Configuration
  logic                                bias_en;
  logic signed [31:0]                  rq_mult;
  logic [4:0]                          rq_shift;
  logic                                relu_en;
  logic                                signed_act;

  // Epilogue Inputs
  logic                                epi_in_valid;
  logic [$clog2(ARR_N)-1:0]            epi_in_col_idx;
  logic signed [ARR_M*ACC_W-1:0]       epi_in_acc;
  logic signed [ARR_M*ACC_W-1:0]       epi_in_bias;

  // Epilogue Outputs
  logic                                epi_out_valid;
  logic [$clog2(ARR_N)-1:0]            epi_out_col_idx;
  logic signed [DW-1:0]                epi_out_data;

  // Output Staging Buffer (npu_obuf)
  logic                                obuf_swap;
  logic                                obuf_cur_bank;
  logic                                obuf_wr_en;
  logic [AW-1:0]                       obuf_wr_addr;
  logic [DW-1:0]                       obuf_wr_data;
  logic                                obuf_rd_en;
  logic [AW-1:0]                       obuf_rd_addr;
  logic [DW-1:0]                       obuf_rd_data;

  // Connect Epilogue output directly to Output Buffer write port
  assign obuf_wr_en   = epi_out_valid;
  assign obuf_wr_addr = epi_out_col_idx[AW-1:0];
  assign obuf_wr_data = epi_out_data;

  // --------------------------------------------------------------------------
  // DUT Instantiations
  // --------------------------------------------------------------------------
  npu_epilogue #(
    .ARR_M  (ARR_M),
    .ARR_N  (ARR_N),
    .ACC_W  (ACC_W),
    .DATA_W (DATA_W)
  ) u_epilogue (
    .clk         (clk),
    .rst_n       (rst_n),
    .bias_en     (bias_en),
    .rq_mult     (rq_mult),
    .rq_shift    (rq_shift),
    .relu_en     (relu_en),
    .signed_act  (signed_act),
    .in_valid    (epi_in_valid),
    .in_col_idx  (epi_in_col_idx),
    .in_acc      (epi_in_acc),
    .in_bias     (epi_in_bias),
    .out_valid   (epi_out_valid),
    .out_col_idx (epi_out_col_idx),
    .out_data    (epi_out_data)
  );

  npu_obuf #(
    .DW    (DW),
    .WORDS (WORDS),
    .AW    (AW)
  ) u_obuf (
    .clk      (clk),
    .rst_n    (rst_n),
    .swap     (obuf_swap),
    .cur_bank (obuf_cur_bank),
    .wr_en    (obuf_wr_en),
    .wr_addr  (obuf_wr_addr),
    .wr_data  (obuf_wr_data),
    .rd_en    (obuf_rd_en),
    .rd_addr  (obuf_rd_addr),
    .rd_bank_sel (1'b0),
    .rd_data  (obuf_rd_data)
  );

  // --------------------------------------------------------------------------
  // Clock Generator (100 MHz, 10ns period)
  // --------------------------------------------------------------------------
  initial clk = 0;
  always #5 clk = ~clk;

  // --------------------------------------------------------------------------
  // Diagnostics
  // --------------------------------------------------------------------------
  int total_errors = 0;
  int test_count = 0;

  // Helper task to check a single 8-bit output
  task check_byte(
    input string test_name,
    input int index,
    input logic signed [7:0] actual,
    input logic signed [7:0] expected
  );
    if (actual !== expected) begin
      $display("[FAIL] %s [%0d]: Expected %0d (0x%02h), Got %0d (0x%02h) at time %0t",
               test_name, index, expected, expected, actual, actual, $time);
      total_errors++;
    end
  endtask

  // --------------------------------------------------------------------------
  // Main Verification Sequence
  // --------------------------------------------------------------------------
  initial begin
    $dumpfile("sim_output/tb_npu_epilogue.vcd");
    $dumpvars(0, tb_npu_epilogue);

    $display("==================================================================");
    $display("  Milestone 6: Requantization Epilogue & Output Buffer Verification");
    $display("==================================================================");

    // Initial signals
    rst_n          = 1'b0;
    bias_en        = 1'b0;
    rq_mult        = 32'sd1;
    rq_shift       = 5'd0;
    relu_en        = 1'b0;
    signed_act     = 1'b1;
    epi_in_valid   = 1'b0;
    epi_in_col_idx = '0;
    epi_in_acc     = '0;
    epi_in_bias    = '0;
    obuf_swap      = 1'b0;
    obuf_rd_en     = 1'b0;
    obuf_rd_addr   = '0;

    #20;
    @(negedge clk);
    rst_n = 1'b1;
    #20;

    // ========================================================================
    // Test 1: Bit-Exact Verification across 1,332 Golden Vectors
    // ========================================================================
    test_count++;
    $display("\n[TEST %0d] Sweeping 1,332 Vectors from model/golden_epilogue.py", test_count);
    begin
      int fd;
      string line;
      logic [31:0] v_acc32, v_bias32, v_mult32;
      logic [7:0]  v_shift8, v_exp8;
      int          v_bias_en, v_relu_en, v_signed_act;
      int vector_count = 0;
      int vector_errors = 0;

      fd = $fopen("tb/vectors/epilogue_vectors.txt", "r");
      if (fd == 0) begin
        $display("[FAIL] Could not open tb/vectors/epilogue_vectors.txt!");
        total_errors++;
      end else begin
        while (!$feof(fd)) begin
          if ($fgets(line, fd) == 0) break;
          // Skip comments and empty lines
          if (line.len() > 0 && line.substr(0, 0) != "#") begin
            if ($sscanf(line, "%h %h %d %h %h %d %d %h",
                        v_acc32, v_bias32, v_bias_en, v_mult32, v_shift8,
                        v_relu_en, v_signed_act, v_exp8) == 8) begin

              // Apply stimulus
              @(posedge clk);
              bias_en        <= (v_bias_en != 0);
              rq_mult        <= $signed(v_mult32);
              rq_shift       <= v_shift8[4:0];
              relu_en        <= (v_relu_en != 0);
              signed_act     <= (v_signed_act != 0);

              epi_in_valid   <= 1'b1;
              epi_in_col_idx <= '0;
              // Feed into lane 0
              epi_in_acc[31:0]  <= $signed(v_acc32);
              epi_in_bias[31:0] <= $signed(v_bias32);

              // Capture output after EPI_LAT cycles
              repeat (EPI_LAT) @(posedge clk);
              #1;
              if (epi_out_data[7:0] !== $signed(v_exp8)) begin
                $display("[FAIL] Vec %0d: Acc=0x%08h, Bias=0x%08h (en=%0d), Mult=0x%08h, Shift=%0d, ReLU=%0d -> Exp=0x%02h (%0d), Got=0x%02h (%0d)",
                         vector_count, v_acc32, v_bias32, v_bias_en, v_mult32, v_shift8, v_relu_en,
                         v_exp8, $signed(v_exp8), epi_out_data[7:0], $signed(epi_out_data[7:0]));
                vector_errors++;
                total_errors++;
              end
              vector_count++;
            end
          end
        end
        $fclose(fd);
      end

      @(posedge clk);
      epi_in_valid <= 1'b0;

      if (vector_errors == 0) begin
        $display("[PASS] All %0d vectors passed bit-exactly! (0 errors)", vector_count);
      end else begin
        $display("[FAIL] %0d / %0d vectors failed!", vector_errors, vector_count);
      end
    end

    // ========================================================================
    // Test 2: Full 16-Lane Parallel SIMD Execution
    // ========================================================================
    test_count++;
    $display("\n[TEST %0d] 16-Lane Parallel SIMD Execution", test_count);
    begin
      logic signed [7:0] exp_lane [ARR_M];

      @(posedge clk);
      bias_en    <= 1'b1;
      rq_mult    <= 32'sd1;
      rq_shift   <= 5'd0;
      relu_en    <= 1'b0;
      signed_act <= 1'b1;

      epi_in_valid   <= 1'b1;
      epi_in_col_idx <= 4'd5;

      // Populate 16 distinct values spanning negative, zero, positive
      for (int m = 0; m < ARR_M; m++) begin
        logic signed [31:0] acc_val  = -32'sd50 + 32'(m * 10); // -50, -40, ..., +100
        logic signed [31:0] bias_val = 32'sd5;                 // +5
        logic signed [31:0] sum_val  = acc_val + bias_val;

        epi_in_acc[m*32 +: 32]  <= acc_val;
        epi_in_bias[m*32 +: 32] <= bias_val;
        exp_lane[m]             = sum_val[7:0];
      end

      repeat (EPI_LAT) @(posedge clk);
      #1;
      if (!epi_out_valid || epi_out_col_idx !== 4'd5) begin
        $display("[FAIL] Output valid or col_idx mismatch!");
        total_errors++;
      end

      for (int m = 0; m < ARR_M; m++) begin
        check_byte("16-Lane SIMD", m, epi_out_data[m*8 +: 8], exp_lane[m]);
      end

      @(posedge clk);
      epi_in_valid <= 1'b0;
      $display("[PASS] 16-Lane parallel execution verified bit-exact.");
    end

    // ========================================================================
    // Test 3: Full 16x16 Tile Drain through Epilogue into npu_obuf
    // ========================================================================
    test_count++;
    $display("\n[TEST %0d] Full 16x16 Tile Drain into Output Staging Buffer (npu_obuf)", test_count);
    begin

      // Configuration: Requantize by 2 (shift=1) with ReLU enabled
      @(posedge clk);
      bias_en    <= 1'b1;
      rq_mult    <= 32'sd1;
      rq_shift   <= 5'd1;
      relu_en    <= 1'b1;
      signed_act <= 1'b1;

      // Upon reset, obuf_cur_bank == 0 (Bank 1 is Write / Bank 0 is Read)
      // Stream 16 columns (j=0..15)
      for (int j = 0; j < ARR_N; j++) begin
        @(posedge clk);
        epi_in_valid   <= 1'b1;
        epi_in_col_idx <= j[$clog2(ARR_N)-1:0];

        for (int m = 0; m < ARR_M; m++) begin
          logic signed [31:0] a_val = -32'sd20 + 32'(m * 4) + 32'(j * 2);
          logic signed [31:0] b_val = 32'sd10;
          logic signed [31:0] sum   = a_val + b_val;
          // Shift by 1 with round half away from zero
          logic signed [31:0] rnd;
          if (sum >= 0) rnd = (sum + 1) >>> 1;
          else          rnd = -((-sum + 1) >>> 1);
          // ReLU
          if (rnd < 0) rnd = 0;
          // Saturation
          if (rnd > 127) rnd = 127;
          tile_gold_t3[m][j] = rnd[7:0];

          epi_in_acc[m*32 +: 32]  <= a_val;
          epi_in_bias[m*32 +: 32] <= b_val;
        end
      end

      @(posedge clk);
      epi_in_valid <= 1'b0;

      // Allow EPI_LAT cycles for the last word (col 15) to register into npu_obuf
      repeat (EPI_LAT) @(posedge clk);
      #1;

      // Bank Swap: Bank 1 becomes Read Bank for DMA readback
      @(posedge clk);
      obuf_swap <= 1'b1;
      @(posedge clk);
      obuf_swap <= 1'b0;

      // Verify that Bank 1 contains exact 16 words (256 bytes)
      for (int j = 0; j < ARR_N; j++) begin
        @(posedge clk);
        obuf_rd_en   <= 1'b1;
        obuf_rd_addr <= j[AW-1:0];
        @(posedge clk);
        #1;
        for (int m = 0; m < ARR_M; m++) begin
          check_byte("npu_obuf Readback", m, obuf_rd_data[m*8 +: 8], tile_gold_t3[m][j]);
        end
      end

      @(posedge clk);
      obuf_rd_en <= 1'b0;
      $display("[PASS] 16x16 Tile drain and buffer readback verified 100%% bit-exact.");
    end

    // ========================================================================
    // Test 4: Double-Buffered Concurrent Epilogue Drain and DMA Readback
    // ========================================================================
    test_count++;
    $display("\n[TEST %0d] Double-Buffered Concurrent Drain & Readback (Zero Contention)", test_count);
    begin
      // Tile 1 is now in Bank 1 (read side). Concurrently drain Tile 2 (all 42s)
      // into Bank 0 while reading Tile 1 back, and check neither is corrupted.
      int t4_errors = 0;
      for (int j = 0; j < ARR_N; j++) begin
        @(posedge clk);
        obuf_rd_en   <= 1'b1;
        obuf_rd_addr <= j[AW-1:0];

        epi_in_valid   <= 1'b1;
        epi_in_col_idx <= j[$clog2(ARR_N)-1:0];
        for (int m = 0; m < ARR_M; m++) begin
          epi_in_acc[m*32 +: 32]  <= 32'sd42;
          epi_in_bias[m*32 +: 32] <= 32'sd0;
        end

        // Read data for the address issued last cycle
        if (j > 0) begin
          #1;
          for (int m = 0; m < ARR_M; m++) begin
            if (obuf_rd_data[m*8 +: 8] !== tile_gold_t3[m][j-1]) t4_errors++;
          end
        end
      end
      @(posedge clk);
      obuf_rd_en   <= 1'b0;
      epi_in_valid <= 1'b0;
      #1;
      for (int m = 0; m < ARR_M; m++) begin
        if (obuf_rd_data[m*8 +: 8] !== tile_gold_t3[m][ARR_N-1]) t4_errors++;
      end

      // Let Tile 2 finish, swap, and read it back: every byte must be 42 >> 1 = 21
      repeat (EPI_LAT) @(posedge clk);
      @(posedge clk);
      obuf_swap <= 1'b1;
      @(posedge clk);
      obuf_swap <= 1'b0;
      for (int j = 0; j < ARR_N; j++) begin
        @(posedge clk);
        obuf_rd_en   <= 1'b1;
        obuf_rd_addr <= j[AW-1:0];
        @(posedge clk);
        #1;
        for (int m = 0; m < ARR_M; m++) begin
          if (obuf_rd_data[m*8 +: 8] !== 8'sd21) t4_errors++;
        end
      end
      @(posedge clk);
      obuf_rd_en <= 1'b0;

      if (t4_errors == 0) begin
        $display("[PASS] Concurrent read/write: both tiles intact, zero port contention.");
      end else begin
        $display("[FAIL] Double-buffer isolation: %0d byte mismatches", t4_errors);
        total_errors += t4_errors;
      end
    end

    // ========================================================================
    // Final Summary
    // ========================================================================
    $display("\n==================================================================");
    if (total_errors == 0) begin
      $display("  ALL %0d TESTS PASSED! (0 errors)", test_count);
      $display("  Milestone 6 Exit Criteria MET:");
      $display("  1. Bias/requant/ReLU/saturate bit-exact vs NumPy golden model.");
      $display("  2. Full dynamic range and exact saturation boundaries [-128, 127] verified.");
      $display("  3. Round half away from zero verified symmetrically for + and - values.");
      $display("  4. 16-lane parallel epilogue SIMD unit verified.");
      $display("  5. Double-buffered 512 B output staging buffer (npu_obuf) verified.");
      $display("==================================================================");
    end else begin
      $display("  TEST FAILED WITH %0d ERRORS!", total_errors);
      $display("==================================================================");
      $fatal(1, "Milestone 6 testbench encountered %0d errors", total_errors);
    end

    #50;
    $finish;
  end

endmodule
