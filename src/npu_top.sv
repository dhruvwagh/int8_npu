// ============================================================================
// File: npu_top.sv
// Description: INT8 Systolic-Array NPU Top Level
//              - AXI4-Lite slave for configuration / control / perf counters
//              - One MEM_DW-wide tensor memory master port shared by weight,
//                activation and bias fetch plus output writeback
//              - Datapath: fetch -> ping-pong buffers -> skew -> ARR_M x ARR_N
//                output-stationary array -> drain -> epilogue -> obuf -> writeback
//
// Array orientation: row i of the array computes output channel m0+i (fed from
// the weight buffer, west edge); column j computes output position n0+j (fed
// from the activation buffer, north edge). PE(i,j) accumulates C[m0+i][n0+j].
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_TOP_SV
`define NPU_TOP_SV

`timescale 1ns/1ps

module npu_top #(
  parameter int ARR_M      = 16,
  parameter int ARR_N      = 16,
  parameter int ACC_W      = 32,
  parameter int K_TILE_MAX = 128,
  parameter int ADDR_W     = 32,
  parameter int MEM_DW     = 128
) (
  input  logic                clk,
  input  logic                rst_n,

  // AXI4-Lite Slave (CSRs)
  input  logic                s_axil_awvalid,
  output logic                s_axil_awready,
  input  logic [7:0]          s_axil_awaddr,
  input  logic                s_axil_wvalid,
  output logic                s_axil_wready,
  input  logic [31:0]         s_axil_wdata,
  input  logic [3:0]          s_axil_wstrb,
  output logic                s_axil_bvalid,
  input  logic                s_axil_bready,
  output logic [1:0]          s_axil_bresp,
  input  logic                s_axil_arvalid,
  output logic                s_axil_arready,
  input  logic [7:0]          s_axil_araddr,
  output logic                s_axil_rvalid,
  input  logic                s_axil_rready,
  output logic [31:0]         s_axil_rdata,
  output logic [1:0]          s_axil_rresp,

  // Tensor Memory Master (in-order read responses)
  output logic                mem_req,
  output logic                mem_we,
  output logic [ADDR_W-1:0]   mem_addr,
  output logic [MEM_DW-1:0]   mem_wdata,
  input  logic                mem_ready,
  input  logic                mem_rvalid,
  input  logic [MEM_DW-1:0]   mem_rdata,

  output logic                irq
);

  localparam int DATA_W = 8;
  localparam int KAW    = $clog2(K_TILE_MAX);
  localparam int OAW    = $clog2(ARR_N);
  localparam int BIAS_WORDS = (ARR_M * 32) / MEM_DW;

  initial begin
    if (MEM_DW != ARR_M * DATA_W) $fatal(1, "npu_top: MEM_DW must equal ARR_M*8 (one panel word per beat)");
  end

  // --------------------------------------------------------------------------
  // CSR block
  // --------------------------------------------------------------------------
  logic [15:0]        dim_m, dim_n, dim_k;
  logic [ADDR_W-1:0]  a_base, b_base, c_base, bias_base;
  logic signed [31:0] rq_mult;
  logic [4:0]         rq_shift;
  logic               relu_en, bias_en, signed_act;
  logic               csr_start, csr_abort, csr_perf_clr;
  logic               seq_busy, seq_done, seq_err;
  logic [31:0]        perf_cycles, perf_macs, perf_stall;

  npu_csr #(
    .ADDR_W(ADDR_W), .ARR_M(ARR_M), .ARR_N(ARR_N),
    .K_TILE_MAX(K_TILE_MAX), .MEM_DW(MEM_DW)
  ) u_csr (
    .clk, .rst_n,
    .s_axil_awvalid, .s_axil_awready, .s_axil_awaddr,
    .s_axil_wvalid,  .s_axil_wready,  .s_axil_wdata, .s_axil_wstrb,
    .s_axil_bvalid,  .s_axil_bready,  .s_axil_bresp,
    .s_axil_arvalid, .s_axil_arready, .s_axil_araddr,
    .s_axil_rvalid,  .s_axil_rready,  .s_axil_rdata, .s_axil_rresp,
    .dim_m, .dim_n, .dim_k, .a_base, .b_base, .c_base, .bias_base,
    .rq_mult, .rq_shift, .relu_en, .bias_en, .signed_act,
    .start(csr_start), .abort(csr_abort), .perf_clr(csr_perf_clr),
    .busy(seq_busy), .done_pulse(seq_done), .err_pulse(seq_err),
    .perf_cycles, .perf_macs, .perf_stall,
    .irq
  );

  // --------------------------------------------------------------------------
  // Sequencer
  // --------------------------------------------------------------------------
  logic              fetch_wgt_start, fetch_wgt_busy, fetch_wgt_done;
  logic [ADDR_W-1:0] fetch_wgt_base_addr;
  logic [7:0]        fetch_wgt_words;
  logic              fetch_act_start, fetch_act_busy, fetch_act_done;
  logic [ADDR_W-1:0] fetch_act_base_addr;
  logic [7:0]        fetch_act_words;
  logic              fetch_bias_start, fetch_bias_busy, fetch_bias_done;
  logic [ADDR_W-1:0] fetch_bias_base_addr;
  logic              wgt_buf_swap, act_buf_swap, wgt_cur_bank, act_cur_bank;
  logic              wgt_rd_en, act_rd_en;
  logic [KAW-1:0]    wgt_rd_addr, act_rd_addr;
  logic              array_en, clr_acc, drain_start, drain_done, drain_ready;
  logic [ADDR_W-1:0] drain_c_addr;
  logic [7:0]        drain_n_len;
  logic [ARR_M-1:0]  active_m_mask;
  logic [ARR_N-1:0]  active_n_mask;
  logic              tile_valid;
  npu_pkg::tile_desc_t tile_desc;

  npu_sequencer #(
    .ARR_M(ARR_M), .ARR_N(ARR_N), .K_TILE_MAX(K_TILE_MAX),
    .ADDR_W(ADDR_W), .DATA_W(DATA_W)
  ) u_seq (
    .clk, .rst_n,
    .seq_start(csr_start), .seq_abort(csr_abort),
    .dim_m, .dim_n, .dim_k, .a_base, .b_base, .c_base, .bias_base,
    .seq_busy, .seq_done, .seq_err,
    .perf_cycles, .perf_macs, .perf_stall,
    .perf_clr(csr_perf_clr),
    .fetch_wgt_start, .fetch_wgt_base_addr, .fetch_wgt_words,
    .fetch_wgt_busy, .fetch_wgt_done,
    .fetch_act_start, .fetch_act_base_addr, .fetch_act_words,
    .fetch_act_busy, .fetch_act_done,
    .fetch_bias_start, .fetch_bias_base_addr,
    .fetch_bias_busy, .fetch_bias_done,
    .wgt_buf_swap, .act_buf_swap,
    .wgt_buf_cur_bank(wgt_cur_bank), .act_buf_cur_bank(act_cur_bank),
    .wgt_rd_en, .wgt_rd_addr, .act_rd_en, .act_rd_addr,
    .array_en, .clr_acc, .drain_start, .drain_done, .drain_ready,
    .drain_c_addr, .drain_n_len,
    .tile_valid, .tile_desc,
    .active_m_mask, .active_n_mask
  );

  // --------------------------------------------------------------------------
  // Memory arbiter: 0 = writeback, 1 = weights, 2 = activations, 3 = bias
  // --------------------------------------------------------------------------
  localparam int NREQ = 4;
  logic [NREQ-1:0]        arb_req, arb_we, arb_ready, arb_rsp_valid;
  logic [NREQ*ADDR_W-1:0] arb_addr;
  logic [NREQ*MEM_DW-1:0] arb_wdata;
  logic [MEM_DW-1:0]      arb_rsp_rdata;

  npu_mem_arb #(
    .NREQ(NREQ), .MEM_DW(MEM_DW), .ADDR_W(ADDR_W)
  ) u_arb (
    .clk, .rst_n,
    .req(arb_req), .req_we(arb_we), .req_addr(arb_addr), .req_wdata(arb_wdata),
    .req_ready(arb_ready), .rsp_valid(arb_rsp_valid), .rsp_rdata(arb_rsp_rdata),
    .mem_req, .mem_we, .mem_addr, .mem_wdata, .mem_ready, .mem_rvalid, .mem_rdata
  );

  assign arb_we = 4'b0001;
  assign arb_wdata[1*MEM_DW +: 3*MEM_DW] = '0;

  // --------------------------------------------------------------------------
  // Fetch engines and operand buffers
  // --------------------------------------------------------------------------
  logic              wgt_buf_wr_en, act_buf_wr_en, bias_wr_en;
  logic [KAW-1:0]    wgt_buf_wr_addr, act_buf_wr_addr;
  logic [MEM_DW-1:0] wgt_buf_wr_data, act_buf_wr_data, bias_wr_data;
  logic [$clog2(BIAS_WORDS)-1:0] bias_wr_addr;
  logic [MEM_DW-1:0] wgt_rd_data, act_rd_data;

  npu_fetch #(.MEM_DW(MEM_DW), .ADDR_W(ADDR_W), .WORDS(K_TILE_MAX)) u_fetch_wgt (
    .clk, .rst_n,
    .fetch_start(fetch_wgt_start), .fetch_base_addr(fetch_wgt_base_addr),
    .fetch_words(fetch_wgt_words), .fetch_busy(fetch_wgt_busy), .fetch_done(fetch_wgt_done),
    .mem_req(arb_req[1]), .mem_addr(arb_addr[1*ADDR_W +: ADDR_W]),
    .mem_ready(arb_ready[1]), .mem_valid(arb_rsp_valid[1]), .mem_rdata(arb_rsp_rdata),
    .buf_wr_en(wgt_buf_wr_en), .buf_wr_addr(wgt_buf_wr_addr), .buf_wr_data(wgt_buf_wr_data)
  );

  npu_fetch #(.MEM_DW(MEM_DW), .ADDR_W(ADDR_W), .WORDS(K_TILE_MAX)) u_fetch_act (
    .clk, .rst_n,
    .fetch_start(fetch_act_start), .fetch_base_addr(fetch_act_base_addr),
    .fetch_words(fetch_act_words), .fetch_busy(fetch_act_busy), .fetch_done(fetch_act_done),
    .mem_req(arb_req[2]), .mem_addr(arb_addr[2*ADDR_W +: ADDR_W]),
    .mem_ready(arb_ready[2]), .mem_valid(arb_rsp_valid[2]), .mem_rdata(arb_rsp_rdata),
    .buf_wr_en(act_buf_wr_en), .buf_wr_addr(act_buf_wr_addr), .buf_wr_data(act_buf_wr_data)
  );

  npu_fetch #(.MEM_DW(MEM_DW), .ADDR_W(ADDR_W), .WORDS(BIAS_WORDS)) u_fetch_bias (
    .clk, .rst_n,
    .fetch_start(fetch_bias_start), .fetch_base_addr(fetch_bias_base_addr),
    .fetch_words(8'(BIAS_WORDS)), .fetch_busy(fetch_bias_busy), .fetch_done(fetch_bias_done),
    .mem_req(arb_req[3]), .mem_addr(arb_addr[3*ADDR_W +: ADDR_W]),
    .mem_ready(arb_ready[3]), .mem_valid(arb_rsp_valid[3]), .mem_rdata(arb_rsp_rdata),
    .buf_wr_en(bias_wr_en), .buf_wr_addr(bias_wr_addr), .buf_wr_data(bias_wr_data)
  );

  npu_buf_wgt #(.DW(MEM_DW), .WORDS(K_TILE_MAX)) u_buf_wgt (
    .clk, .rst_n,
    .swap(wgt_buf_swap), .cur_bank(wgt_cur_bank),
    .rd_en(wgt_rd_en), .rd_addr(wgt_rd_addr), .rd_data(wgt_rd_data),
    .wr_en(wgt_buf_wr_en), .wr_addr(wgt_buf_wr_addr), .wr_data(wgt_buf_wr_data)
  );

  npu_buf_act #(.DW(MEM_DW), .WORDS(K_TILE_MAX)) u_buf_act (
    .clk, .rst_n,
    .swap(act_buf_swap), .cur_bank(act_cur_bank),
    .rd_en(act_rd_en), .rd_addr(act_rd_addr), .rd_data(act_rd_data),
    .wr_en(act_buf_wr_en), .wr_addr(act_buf_wr_addr), .wr_data(act_buf_wr_data)
  );

  // Per-output-channel bias for the current M panel (lane i at [32i +: 32])
  logic [ARR_M*32-1:0] bias_regs;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bias_regs <= '0;
    end else if (bias_wr_en) begin
      bias_regs[bias_wr_addr*MEM_DW +: MEM_DW] <= bias_wr_data;
    end
  end

  // --------------------------------------------------------------------------
  // Operand alignment: buffer reads have 1-cycle latency, so the data valid
  // flag and the accumulator clear are delayed by one cycle to line up with
  // the first word. Outside a valid beat the array is fed zeros, which is what
  // flushes the systolic pipeline during fill.
  // --------------------------------------------------------------------------
  logic operand_valid, clr_aligned;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      operand_valid <= 1'b0;
      clr_aligned   <= 1'b0;
    end else begin
      operand_valid <= wgt_rd_en;
      clr_aligned   <= clr_acc;
    end
  end

  logic signed [ARR_M*DATA_W-1:0] row_operands;
  logic signed [ARR_N*DATA_W-1:0] col_operands;
  always_comb begin
    for (int i = 0; i < ARR_M; i++) begin
      row_operands[i*DATA_W +: DATA_W] = (operand_valid && active_m_mask[i]) ? wgt_rd_data[i*DATA_W +: DATA_W] : '0;
    end
    for (int j = 0; j < ARR_N; j++) begin
      col_operands[j*DATA_W +: DATA_W] = (operand_valid && active_n_mask[j]) ? act_rd_data[j*DATA_W +: DATA_W] : '0;
    end
  end

  // --------------------------------------------------------------------------
  // Skew + systolic array + drain
  // --------------------------------------------------------------------------
  logic signed [ARR_M*DATA_W-1:0]      a_skewed;
  logic signed [ARR_N*DATA_W-1:0]      b_skewed;
  logic        [ARR_M*ARR_N-1:0]       clr_skewed;
  logic signed [ARR_M*ARR_N*ACC_W-1:0] acc;
  logic signed [ARR_M*DATA_W-1:0]      a_east_unused;
  logic signed [ARR_N*DATA_W-1:0]      b_south_unused;

  // The skew and array free-run (HAS_EN = 0). Idle cycles feed zero operands,
  // so accumulators hold (acc += 0) without a 12K-fanout enable net.
  npu_skew #(.ARR_M(ARR_M), .ARR_N(ARR_N), .DATA_W(DATA_W), .HAS_EN(1'b0)) u_skew (
    .clk, .rst_n, .en(1'b1), .clr_in(clr_aligned),
    .a_in(row_operands), .b_in(col_operands),
    .a_skewed, .b_skewed, .clr_skewed
  );

  npu_array #(.ARR_M(ARR_M), .ARR_N(ARR_N), .ACC_W(ACC_W), .HAS_EN(1'b0)) u_array (
    .clk, .rst_n, .en(1'b1), .clr(clr_skewed),
    .a_in(a_skewed), .b_in(b_skewed),
    .a_out(a_east_unused), .b_out(b_south_unused), .acc
  );

  logic                          drain_busy, drain_valid;
  logic [OAW-1:0]                drain_col_idx;
  logic signed [ARR_M*ACC_W-1:0] drain_data;

  npu_drain #(.ARR_M(ARR_M), .ARR_N(ARR_N), .ACC_W(ACC_W)) u_drain (
    .clk, .rst_n, .drain_start, .acc,
    .drain_busy, .drain_valid, .drain_col_idx, .drain_data, .drain_done
  );

  // --------------------------------------------------------------------------
  // Epilogue -> output buffer -> writeback
  // --------------------------------------------------------------------------
  logic                        epi_valid;
  logic [OAW-1:0]              epi_col_idx;
  logic signed [MEM_DW-1:0]    epi_data;

  npu_epilogue #(.ARR_M(ARR_M), .ARR_N(ARR_N), .ACC_W(ACC_W), .DATA_W(DATA_W)) u_epi (
    .clk, .rst_n,
    .bias_en, .rq_mult, .rq_shift, .relu_en, .signed_act,
    .in_valid(drain_valid), .in_col_idx(drain_col_idx),
    .in_acc(drain_data), .in_bias(bias_regs),
    .out_valid(epi_valid), .out_col_idx(epi_col_idx), .out_data(epi_data)
  );

  // Swap output banks as the last column is written; writeback starts next cycle
  wire obuf_swap = epi_valid && (epi_col_idx == OAW'(ARR_N - 1));

  logic              obuf_rd_en;
  logic [OAW-1:0]    obuf_rd_addr;
  logic [MEM_DW-1:0] obuf_rd_data;
  logic              obuf_cur_bank;

  npu_obuf #(.DW(MEM_DW), .WORDS(ARR_N)) u_obuf (
    .clk, .rst_n,
    .swap(obuf_swap), .cur_bank(obuf_cur_bank),
    .wr_en(epi_valid), .wr_addr(epi_col_idx), .wr_data(epi_data),
    .rd_en(obuf_rd_en), .rd_addr(obuf_rd_addr), .rd_data(obuf_rd_data)
  );

  // A drained tile is "in flight" from drain_start until its writeback starts.
  // The sequencer may only start the next drain once that tile has fully
  // left the obuf read bank.
  logic wb_start, wb_busy, drain_inflight;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wb_start       <= 1'b0;
      drain_inflight <= 1'b0;
    end else begin
      wb_start <= obuf_swap;
      if (drain_start) drain_inflight <= 1'b1;
      else if (wb_start) drain_inflight <= 1'b0;
    end
  end
  assign drain_ready = !drain_inflight && !wb_busy && !wb_start;

  npu_wb #(.MEM_DW(MEM_DW), .ADDR_W(ADDR_W), .WORDS(ARR_N)) u_wb (
    .clk, .rst_n,
    .wb_start, .wb_base_addr(drain_c_addr), .wb_words(drain_n_len), .wb_busy,
    .obuf_rd_en, .obuf_rd_addr, .obuf_rd_data,
    .mem_req(arb_req[0]), .mem_addr(arb_addr[0 +: ADDR_W]),
    .mem_wdata(arb_wdata[0 +: MEM_DW]), .mem_ready(arb_ready[0])
  );

  // Debug-only outputs left unconnected at the top level
  logic _unused;
  assign _unused = &{1'b0, tile_valid, tile_desc, drain_busy, obuf_cur_bank, arb_rsp_valid[0], array_en,
                     a_east_unused, b_south_unused, 1'b0};

endmodule : npu_top

`endif // NPU_TOP_SV
