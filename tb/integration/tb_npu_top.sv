// ============================================================================
// File: tb_npu_top.sv
// Description: End-to-end testbench for npu_top
//              - AXI4-Lite master tasks program CSRs, start a layer, poll STATUS
//              - Tensor memory model: in-order reads with configurable latency,
//                random jitter and random backpressure (mem_ready)
//              - After each layer the WHOLE memory image is compared against the
//                golden image from model/golden_npu.py, so wrong outputs, missed
//                writes and stray writes are all caught
//              - Also checks perf counters, CSR readback, SLVERR, and M=0 error
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module tb_npu_top;

  import npu_pkg::*;

  `include "npu_tb_harness.svh"

  logic [MEM_DW-1:0] mem_exp [MEM_WORDS];

  // --------------------------------------------------------------------------
  // Directed CSR checks
  // --------------------------------------------------------------------------
  task automatic test_csr();
    logic [31:0] v;
    logic [1:0]  resp;
    int errs = 0;
    logic [7:0]  regs [10] = '{CSR_DIM_M, CSR_DIM_N, CSR_DIM_K, CSR_A_BASE, CSR_B_BASE,
                               CSR_C_BASE, CSR_BIAS_BASE, CSR_RQ_MULT, CSR_RQ_SHIFT, CSR_FLAGS};
    logic [31:0] vals [10] = '{32'h0000_1234, 32'h0000_00F0, 32'h0000_0310, 32'hDEAD_BEE0,
                               32'h1234_5670, 32'h0BAD_F00D, 32'hCAFE_0040, 32'h8000_0001,
                               32'h0000_001F, 32'h0000_0007};
    $display("\n[TEST] CSR readback, HW_CFG, SLVERR on unmapped addresses");
    for (int i = 0; i < 10; i++) csr_wr(regs[i], vals[i]);
    for (int i = 0; i < 10; i++) begin
      csr_rd(regs[i], v);
      if (v !== vals[i]) begin
        $display("[FAIL] CSR 0x%02h readback 0x%08h, expected 0x%08h", regs[i], v, vals[i]);
        errs++;
      end
    end
    csr_rd(CSR_HW_CFG, v);
    if (v !== {8'(MEM_DW/8), 8'(K_TILE_MAX/8), 8'(ARR_N), 8'(ARR_M)}) begin
      $display("[FAIL] HW_CFG = 0x%08h", v);
      errs++;
    end
    axil_read(8'h7C, v, resp);
    if (resp !== 2'b10) begin $display("[FAIL] unmapped read resp %0d", resp); errs++; end
    axil_write(8'h7C, 32'h1, resp);
    if (resp !== 2'b10) begin $display("[FAIL] unmapped write resp %0d", resp); errs++; end
    axil_write(CSR_STATUS, 32'h1, resp);
    if (resp !== 2'b10) begin $display("[FAIL] write to read-only STATUS resp %0d", resp); errs++; end
    total_errors += errs;
    if (errs == 0) $display("[PASS] CSR block");
  endtask

  task automatic test_zero_dim();
    logic [31:0] st;
    int errs = 0;
    $display("\n[TEST] M=0 layer raises STATUS.err and does not hang");
    csr_wr(CSR_DIM_M, 0);
    csr_wr(CSR_DIM_N, 4);
    csr_wr(CSR_DIM_K, 4);
    csr_wr(CSR_CTRL, 32'h1);
    repeat (5) @(posedge clk);
    csr_rd(CSR_STATUS, st);
    if (!st[STATUS_ERR_BIT] || st[STATUS_BUSY_BIT] || st[STATUS_DONE_BIT]) begin
      $display("[FAIL] STATUS after M=0 start = 0x%08h", st);
      errs++;
    end
    total_errors += errs;
    if (errs == 0) $display("[PASS] Invalid dimension reported via STATUS.err");
  endtask

  // --------------------------------------------------------------------------
  // Layer test from the golden vector list
  // --------------------------------------------------------------------------
  task automatic run_layer(input int id, input int M, input int N, input int K,
                           input logic [31:0] a_b, input logic [31:0] b_b,
                           input logic [31:0] c_b, input logic [31:0] bias_b,
                           input logic [31:0] mult, input int shift, input int flags,
                           input int nwords, input int exp_macs);
    string f_in, f_exp;
    int ok, errs, shown;
    logic [31:0] cyc, macs, stall;
    real util;

    f_in  = $sformatf("tb/vectors/top/mem_%0d.hex", id);
    f_exp = $sformatf("tb/vectors/top/exp_%0d.hex", id);
    for (int w = 0; w < nwords; w++) begin
      mem[w]     = '0;
      mem_exp[w] = '0;
    end
    $readmemh(f_in, mem, 0, nwords - 1);
    $readmemh(f_exp, mem_exp, 0, nwords - 1);
    mem_writes = 0;
    mem_reads  = 0;

    csr_wr(CSR_DIM_M, M);
    csr_wr(CSR_DIM_N, N);
    csr_wr(CSR_DIM_K, K);
    csr_wr(CSR_A_BASE, a_b);
    csr_wr(CSR_B_BASE, b_b);
    csr_wr(CSR_C_BASE, c_b);
    csr_wr(CSR_BIAS_BASE, bias_b);
    csr_wr(CSR_RQ_MULT, mult);
    csr_wr(CSR_RQ_SHIFT, shift);
    csr_wr(CSR_FLAGS, flags);
    csr_wr(CSR_CTRL, 32'h1);

    // Config registers are write-protected while busy: a write must return
    // SLVERR and must not disturb the running layer (checked bit-exact below)
    errs = 0;
    if (id == 2) begin
      logic [1:0] resp;
      axil_write(CSR_RQ_MULT, 32'h0, resp);
      if (resp !== 2'b10) begin
        $display("[FAIL] RQ_MULT write while busy returned resp %0d, expected SLVERR", resp);
        errs++;
      end
    end

    wait_done(ok);
    if (ok == 0) errs++;
    if (!irq) begin
      $display("[FAIL] irq not asserted after done");
      errs++;
    end

    shown = 0;
    for (int w = 0; w < nwords; w++) begin
      if (mem[w] !== mem_exp[w]) begin
        if (shown < 5) begin
          $display("       word %0d (addr 0x%05h): got %h exp %h", w, w * (MEM_DW / 8), mem[w], mem_exp[w]);
          shown++;
        end
        errs++;
      end
    end
    if (mem_bad_addr != 0) begin
      $display("[FAIL] %0d misaligned/out-of-range memory accesses", mem_bad_addr);
      errs++;
    end

    csr_rd(CSR_PERF_CYCLES, cyc);
    csr_rd(CSR_PERF_MACS, macs);
    csr_rd(CSR_PERF_STALL, stall);
    if (macs != 32'(exp_macs)) begin
      $display("[FAIL] PERF_MACS = %0d, expected %0d", macs, exp_macs);
      errs++;
    end

    util = 100.0 * real'(macs) / (real'(cyc) * ARR_M * ARR_N);
    if (errs == 0) begin
      $display("[PASS] layer %0d  M=%0d N=%0d K=%0d  flags=%0d  lat=%0d+%0d rdy=%0d%%  cycles=%0d stall=%0d util=%0.1f%%",
               id, M, N, K, flags, mem_lat, mem_jit, mem_ready_pct, cyc, stall, util);
    end else begin
      $display("[FAIL] layer %0d  M=%0d N=%0d K=%0d: %0d errors", id, M, N, K, errs);
    end
    total_errors += errs;
  endtask

  // --------------------------------------------------------------------------
  // Main
  // --------------------------------------------------------------------------
  initial begin
    int fd, n_layers, n_pass;
    int id, M, N, K, shift, flags, nwords, lat, jit, rdy, exp_macs;
    logic [31:0] a_b, b_b, c_b, bias_b, mult;

    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
    awaddr = 0; araddr = 0; wdata = 0; wstrb = 0;

    if ($test$plusargs("vcd")) begin
      $dumpfile("sim_output/tb_npu_top.vcd");
      $dumpvars(0, tb_npu_top);
    end

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    test_csr();
    test_zero_dim();

    fd = $fopen("tb/vectors/top/tests.txt", "r");
    if (fd == 0) $fatal(1, "Cannot open tb/vectors/top/tests.txt");
    n_layers = 0;
    while ($fscanf(fd, "%d %d %d %d %h %h %h %h %h %d %d %d %d %d %d %d",
                   id, M, N, K, a_b, b_b, c_b, bias_b, mult, shift, flags,
                   nwords, lat, jit, rdy, exp_macs) == 16) begin
      mem_lat = lat;
      mem_jit = jit;
      mem_ready_pct = rdy;
      run_layer(id, M, N, K, a_b, b_b, c_b, bias_b, mult, shift, flags, nwords, exp_macs);
      n_layers++;
    end
    $fclose(fd);

    $display("\n==================================================================");
    if (total_errors == 0 && n_layers > 0) begin
      $display("  ALL TESTS PASSED: %0d layers bit-exact + CSR + error handling", n_layers);
      $display("==================================================================");
    end else begin
      $display("  FAILED: %0d errors over %0d layers", total_errors, n_layers);
      $display("==================================================================");
      $fatal(1, "tb_npu_top failed");
    end
    $finish;
  end

endmodule : tb_npu_top
