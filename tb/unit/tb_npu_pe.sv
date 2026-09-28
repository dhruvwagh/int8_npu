// ============================================================================
// File: tb_npu_pe.sv
// Description: Unit testbench for npu_pe.sv
//              - Direct sanity & corner cases
//              - Golden vector-file driven validation (bit-exact vs NumPy model)
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module tb_npu_pe;

  localparam int ACC_W = 32;
  localparam string DEFAULT_VECTOR_FILE = "tb/vectors/pe_vectors.hex";

  // Clock & control signals
  logic                     clk;
  logic                     rst_n;
  logic                     en;
  logic                     clr;
  logic signed [7:0]        a_in;
  logic signed [7:0]        b_in;
  wire  signed [7:0]        a_out;
  wire  signed [7:0]        b_out;
  wire  signed [ACC_W-1:0]  acc;

  // DUT instantiation
  npu_pe #(
    .ACC_W(ACC_W)
  ) dut (
    .clk   (clk),
    .rst_n (rst_n),
    .en    (en),
    .clr   (clr),
    .a_in  (a_in),
    .b_in  (b_in),
    .a_out (a_out),
    .b_out (b_out),
    .acc   (acc)
  );

  // 100 MHz clock generation (10 ns period)
  initial clk = 0;
  always #5 clk = ~clk;

  // Testbench bookkeeping
  int vector_count = 0;
  int error_count  = 0;
  int fd;
  string line;
  string vector_file;

  // File read variables
  int rst_n_int, en_int, clr_int;
  logic [7:0]        a_in_raw, b_in_raw;
  logic [7:0]        exp_a_raw, exp_b_raw;
  logic [ACC_W-1:0]  exp_acc_raw;
  logic signed [7:0]        exp_a_out;
  logic signed [7:0]        exp_b_out;
  logic signed [ACC_W-1:0]  exp_acc;

  initial begin
    // Setup waveform dump
    $dumpfile("sim_output/tb_npu_pe.vcd");
    $dumpvars(0, tb_npu_pe);

    // Initial signal values
    rst_n = 1'b0;
    en    = 1'b0;
    clr   = 1'b0;
    a_in  = 8'sd0;
    b_in  = 8'sd0;

    $display("==================================================================");
    $display("  Starting npu_pe Unit Testbench");
    $display("==================================================================");

    // ------------------------------------------------------------------------
    // Part 1: Basic Reset and Directed Sanity Check
    // ------------------------------------------------------------------------
    @(negedge clk);
    rst_n = 1'b0;
    @(posedge clk);
    #1;
    if (a_out !== 8'sd0 || b_out !== 8'sd0 || acc !== '0) begin
      $error("[FAIL] Reset state invalid: a_out=%0d b_out=%0d acc=%0d", a_out, b_out, acc);
      error_count++;
    end else begin
      $display("[PASS] Reset state verified (all zero).");
    end

    // Release reset
    @(negedge clk);
    rst_n = 1'b1;

    // ------------------------------------------------------------------------
    // Part 2: Vector-driven Verification against Golden Model
    // ------------------------------------------------------------------------
    if (!$value$plusargs("VECTORS=%s", vector_file)) begin
      vector_file = DEFAULT_VECTOR_FILE;
    end

    $display("[INFO] Loading golden vectors from: %s", vector_file);
    fd = $fopen(vector_file, "r");
    if (fd == 0) begin
      $error("[ERROR] Failed to open vector file: %s", vector_file);
      $finish(1);
    end

    // Read vector file line by line
    while (!$feof(fd)) begin
      int scan_count;
      string comment_char;

      // Check for comments or empty lines
      scan_count = $fscanf(fd, "%1s", comment_char);
      if (scan_count != 1) break;

      if (comment_char == "#") begin
        // Discard rest of comment line
        void'($fgets(line, fd));
        continue;
      end else begin
        // Push the character back or read using the formatted string
        // Since $ungetc is not universal, we format the scan with the first char
        // Instead, the file has 8 tokens: rst_n en clr a_in b_in exp_a exp_b exp_acc
        // Where comment_char was the first digit of rst_n
        rst_n_int = comment_char.atoi();
        scan_count = $fscanf(fd, "%d %d %x %x %x %x %x\n",
                             en_int, clr_int,
                             a_in_raw, b_in_raw,
                             exp_a_raw, exp_b_raw, exp_acc_raw);
        if (scan_count == 7) begin
          vector_count++;

          // Apply inputs on negedge
          @(negedge clk);
          rst_n = rst_n_int[0];
          en    = en_int[0];
          clr   = clr_int[0];
          a_in  = $signed(a_in_raw);
          b_in  = $signed(b_in_raw);

          exp_a_out = $signed(exp_a_raw);
          exp_b_out = $signed(exp_b_raw);
          exp_acc   = $signed(exp_acc_raw);

          // Check outputs after posedge clock edge + propagation delay
          @(posedge clk);
          #1;

          if (a_out !== exp_a_out) begin
            $error("[MISMATCH][Vec %0d] a_out: got %h (%0d), expected %h (%0d)",
                   vector_count, a_out, a_out, exp_a_out, exp_a_out);
            error_count++;
          end

          if (b_out !== exp_b_out) begin
            $error("[MISMATCH][Vec %0d] b_out: got %h (%0d), expected %h (%0d)",
                   vector_count, b_out, b_out, exp_b_out, exp_b_out);
            error_count++;
          end

          if (acc !== exp_acc) begin
            $error("[MISMATCH][Vec %0d] acc: got %h (%0d), expected %h (%0d)",
                   vector_count, acc, acc, exp_acc, exp_acc);
            error_count++;
          end
        end
      end
    end

    $fclose(fd);

    // ------------------------------------------------------------------------
    // Summary Report
    // ------------------------------------------------------------------------
    $display("==================================================================");
    $display("  PE Unit Test Summary");
    $display("  Total vectors tested : %0d", vector_count);
    $display("  Total errors         : %0d", error_count);
    if (error_count == 0 && vector_count > 0) begin
      $display("  RESULT: *** ALL TESTS PASSED *** (Bit-exact match to NumPy model)");
      $display("==================================================================");
      $finish(0);
    end else begin
      $display("  RESULT: *** TEST FAILED ***");
      $display("==================================================================");
      $finish(1);
    end
  end

endmodule : tb_npu_pe
