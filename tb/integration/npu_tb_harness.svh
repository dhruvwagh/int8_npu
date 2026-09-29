// ============================================================================
// File: npu_tb_harness.svh
// Description: Shared testbench harness for npu_top, `include'd inside a
//              testbench module:
//              - clock/reset, DUT instance
//              - tensor memory model: in-order reads with configurable latency,
//                random jitter and random backpressure
//              - AXI4-Lite master tasks (randomised AW/W ordering), CSR helpers
//              - host byte access to tensor memory
//              Define TB_MEM_WORDS before including to resize memory.
// Author: Dhruv Wagh
// ============================================================================

  localparam int ARR_M      = 16;
  localparam int ARR_N      = 16;
  localparam int K_TILE_MAX = 128;
  localparam int MEM_DW     = 128;
  localparam int ADDR_W     = 32;
`ifndef TB_MEM_WORDS
  `define TB_MEM_WORDS 65536
`endif
  localparam int MEM_WORDS  = `TB_MEM_WORDS;   // tensor memory (16 B words)
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
  // Host-side byte access to tensor memory (zero simulation time)
  // --------------------------------------------------------------------------
  function automatic logic [7:0] mem_get_byte(input logic [31:0] addr);
    logic [MEM_DW-1:0] w;
    w = mem[addr >> 4];
    return w[addr[3:0]*8 +: 8];
  endfunction

  task automatic mem_set_byte(input logic [31:0] addr, input logic [7:0] v);
    logic [MEM_DW-1:0] w;
    w = mem[addr >> 4];
    w[addr[3:0]*8 +: 8] = v;
    mem[addr >> 4] = w;
  endtask

