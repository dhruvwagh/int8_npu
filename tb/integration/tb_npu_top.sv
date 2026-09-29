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

  localparam int ARR_M      = 16;
  localparam int ARR_N      = 16;
  localparam int K_TILE_MAX = 128;
  localparam int MEM_DW     = 128;
  localparam int ADDR_W     = 32;
  localparam int MEM_WORDS  = 65536;           // 1 MB tensor memory
  localparam int RQ_DEPTH   = 64;              // memory model response queue
  localparam int TIMEOUT    = 2_000_000;       // cycles per layer

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  always #5 clk = ~clk;

  // AXI4-Lite
  logic        awvalid, awready, wvalid, wready, bvalid, bready;
  logic [7:0]  awaddr, araddr;
  logic [31:0] wdata, rdata;
  logic [3:0]  wstrb;
  logic [1:0]  bresp, rresp;
  logic        arvalid, arready, rvalid, rready;

  // Memory
  logic              mem_req, mem_we, mem_ready, mem_rvalid;
  logic [ADDR_W-1:0] mem_addr;
  logic [MEM_DW-1:0] mem_wdata, mem_rdata;
  logic              irq;

  npu_top #(
    .ARR_M(ARR_M), .ARR_N(ARR_N), .K_TILE_MAX(K_TILE_MAX),
    .ADDR_W(ADDR_W), .MEM_DW(MEM_DW)
  ) dut (
    .clk, .rst_n,
    .s_axil_awvalid(awvalid), .s_axil_awready(awready), .s_axil_awaddr(awaddr),
    .s_axil_wvalid(wvalid),   .s_axil_wready(wready),   .s_axil_wdata(wdata), .s_axil_wstrb(wstrb),
    .s_axil_bvalid(bvalid),   .s_axil_bready(bready),   .s_axil_bresp(bresp),
    .s_axil_arvalid(arvalid), .s_axil_arready(arready), .s_axil_araddr(araddr),
    .s_axil_rvalid(rvalid),   .s_axil_rready(rready),   .s_axil_rdata(rdata), .s_axil_rresp(rresp),
    .mem_req, .mem_we, .mem_addr, .mem_wdata, .mem_ready, .mem_rvalid, .mem_rdata,
    .irq
  );

  int total_errors = 0;

  // --------------------------------------------------------------------------
  // Tensor memory model
  // --------------------------------------------------------------------------
  logic [MEM_DW-1:0] mem     [MEM_WORDS];
  logic [MEM_DW-1:0] mem_exp [MEM_WORDS];

  int mem_lat       = 2;    // fixed read latency (cycles after accept)
  int mem_jit       = 0;    // extra random latency 0..mem_jit
  int mem_ready_pct = 100;  // probability mem_ready is high in a cycle
  longint cycle = 0;

  logic [MEM_DW-1:0] rq_data [RQ_DEPTH];
  longint            rq_time [RQ_DEPTH];
  int rq_head = 0, rq_tail = 0, rq_cnt = 0;
  longint last_rsp_time = 0;
  int mem_reads = 0, mem_writes = 0, mem_bad_addr = 0;

  always @(posedge clk) begin
    cycle <= cycle + 1;
    if (!rst_n) begin
      mem_ready  <= 1'b0;
      mem_rvalid <= 1'b0;
      rq_head = 0; rq_tail = 0; rq_cnt = 0;
    end else begin
      // Accept a request presented this cycle
      if (mem_req && mem_ready) begin
        if (mem_addr[3:0] != 0 || (mem_addr >> 4) >= MEM_WORDS) begin
          mem_bad_addr++;
        end else if (mem_we) begin
          mem[mem_addr >> 4] = mem_wdata;
          mem_writes++;
        end else begin
          longint t;
          t = cycle + mem_lat + ((mem_jit > 0) ? ($urandom % (mem_jit + 1)) : 0);
          if (t <= last_rsp_time) t = last_rsp_time + 1;   // in order, 1 beat/cycle
          last_rsp_time = t;
          rq_data[rq_tail] = mem[mem_addr >> 4];
          rq_time[rq_tail] = t;
          rq_tail = (rq_tail + 1) % RQ_DEPTH;
          rq_cnt++;
          mem_reads++;
        end
      end

      // Return the oldest read when its time has come
      if (rq_cnt > 0 && rq_time[rq_head] <= cycle) begin
        mem_rvalid <= 1'b1;
        mem_rdata  <= rq_data[rq_head];
        rq_head = (rq_head + 1) % RQ_DEPTH;
        rq_cnt--;
      end else begin
        mem_rvalid <= 1'b0;
        mem_rdata  <= 'x;
      end

      mem_ready <= (($urandom % 100) < mem_ready_pct) && (rq_cnt < RQ_DEPTH - 2);
    end
  end

  // --------------------------------------------------------------------------
  // AXI4-Lite master
  // --------------------------------------------------------------------------
  task automatic axil_write(input logic [7:0] addr, input logic [31:0] data,
                            output logic [1:0] resp);
    int order;
    order = $urandom % 3;   // 0: together, 1: AW first, 2: W first
    @(posedge clk);
    if (order != 2) begin awvalid <= 1'b1; awaddr <= addr; end
    if (order != 1) begin wvalid  <= 1'b1; wdata  <= data; wstrb <= 4'hF; end
    fork
      begin
        if (order == 2) begin @(posedge clk); awvalid <= 1'b1; awaddr <= addr; end
        do @(posedge clk); while (!(awvalid && awready));
        awvalid <= 1'b0;
      end
      begin
        if (order == 1) begin @(posedge clk); wvalid <= 1'b1; wdata <= data; wstrb <= 4'hF; end
        do @(posedge clk); while (!(wvalid && wready));
        wvalid <= 1'b0;
      end
    join
    bready <= 1'b1;
    do @(posedge clk); while (!bvalid);
    resp = bresp;
    bready <= 1'b0;
  endtask

  task automatic axil_read(input logic [7:0] addr, output logic [31:0] data,
                           output logic [1:0] resp);
    @(posedge clk);
    arvalid <= 1'b1;
    araddr  <= addr;
    do @(posedge clk); while (!arready);
    arvalid <= 1'b0;
    rready  <= 1'b1;
    do @(posedge clk); while (!rvalid);
    data = rdata;
    resp = rresp;
    rready <= 1'b0;
  endtask

  task automatic csr_wr(input logic [7:0] addr, input logic [31:0] data);
    logic [1:0] resp;
    axil_write(addr, data, resp);
    if (resp != 2'b00) begin
      $display("[FAIL] CSR write 0x%02h returned resp %0d", addr, resp);
      total_errors++;
    end
  endtask

  task automatic csr_rd(input logic [7:0] addr, output logic [31:0] data);
    logic [1:0] resp;
    axil_read(addr, data, resp);
    if (resp != 2'b00) begin
      $display("[FAIL] CSR read 0x%02h returned resp %0d", addr, resp);
      total_errors++;
    end
  endtask

  task automatic wait_done(output int ok);
    logic [31:0] st;
    int polls = 0;
    ok = 0;
    forever begin
      repeat (20) @(posedge clk);
      csr_rd(CSR_STATUS, st);
      if (st[STATUS_DONE_BIT] || st[STATUS_ERR_BIT]) begin
        ok = st[STATUS_DONE_BIT] && !st[STATUS_ERR_BIT];
        break;
      end
      polls++;
      if (polls * 20 > TIMEOUT) begin
        $display("[FAIL] Timeout waiting for STATUS.done (STATUS=0x%08h)", st);
        break;
      end
    end
  endtask

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

    wait_done(ok);
    errs = (ok == 0) ? 1 : 0;
    if (!irq) begin
      $display("[FAIL] irq not asserted after done");
      errs++;
    end

    shown = 0;
    for (int w = 0; w < nwords; w++) begin
      if (mem[w] !== mem_exp[w]) begin
        if (shown < 5) begin
          $display("       word %0d (addr 0x%05h): got %032h exp %032h", w, w * 16, mem[w], mem_exp[w]);
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
