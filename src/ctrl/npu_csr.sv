// ============================================================================
// File: npu_csr.sv
// Description: AXI4-Lite Slave Control/Status Register Block
//              - 32-bit data, byte addresses per npu_pkg CSR map
//              - One outstanding write and one outstanding read; AW and W may
//                arrive in either order or together
//              - Unmapped addresses return SLVERR (writes are ignored)
//              - CTRL bits are self-clearing pulses; STATUS.done/err are sticky
//                until the next start
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_CSR_SV
`define NPU_CSR_SV

`timescale 1ns/1ps

module npu_csr #(
  parameter int ADDR_W     = 32,
  parameter int ARR_M      = 16,
  parameter int ARR_N      = 16,
  parameter int K_TILE_MAX = 128,
  parameter int MEM_DW     = 128
) (
  input  logic                clk,
  input  logic                rst_n,

  // AXI4-Lite Slave
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

  // Layer configuration
  output logic [15:0]         dim_m,
  output logic [15:0]         dim_n,
  output logic [15:0]         dim_k,
  output logic [ADDR_W-1:0]   a_base,
  output logic [ADDR_W-1:0]   b_base,
  output logic [ADDR_W-1:0]   c_base,
  output logic [ADDR_W-1:0]   bias_base,
  output logic signed [31:0]  rq_mult,
  output logic [4:0]          rq_shift,
  output logic                relu_en,
  output logic                bias_en,
  output logic                signed_act,

  // Control pulses
  output logic                start,
  output logic                abort,
  output logic                perf_clr,

  // Status
  input  logic                busy,
  input  logic                done_pulse,
  input  logic                err_pulse,
  input  logic [31:0]         perf_cycles,
  input  logic [31:0]         perf_macs,
  input  logic [31:0]         perf_stall,
  output logic                irq             // Level: STATUS.done
);

  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_SLVERR = 2'b10;

  logic done_q, err_q;
  logic [31:0] flags_q;

  // Read-only hardware configuration: [7:0] ARR_M, [15:8] ARR_N,
  // [23:16] K_TILE_MAX/8, [31:24] MEM_DW/8
  wire [31:0] hw_cfg = {8'(MEM_DW/8), 8'(K_TILE_MAX/8), 8'(ARR_N), 8'(ARR_M)};

  // --------------------------------------------------------------------------
  // Write channel
  // --------------------------------------------------------------------------
  logic       aw_held, w_held;
  logic [7:0] aw_addr_q;
  logic [31:0] w_data_q;
  logic [3:0]  w_strb_q;

  assign s_axil_awready = !aw_held && !s_axil_bvalid;
  assign s_axil_wready  = !w_held  && !s_axil_bvalid;

  wire aw_fire = s_axil_awvalid && s_axil_awready;
  wire w_fire  = s_axil_wvalid  && s_axil_wready;

  // Address and data for the write about to commit (held or arriving now)
  wire        wr_have_aw = aw_held || aw_fire;
  wire        wr_have_w  = w_held  || w_fire;
  wire [7:0]  wr_addr    = aw_held ? aw_addr_q : s_axil_awaddr;
  wire [31:0] wr_data    = w_held  ? w_data_q  : s_axil_wdata;
  wire [3:0]  wr_strb    = w_held  ? w_strb_q  : s_axil_wstrb;
  wire        wr_commit  = wr_have_aw && wr_have_w && !s_axil_bvalid;

  function automatic logic [31:0] apply_strb(input logic [31:0] old_v,
                                             input logic [31:0] new_v,
                                             input logic [3:0]  strb);
    for (int b = 0; b < 4; b++) begin
      apply_strb[b*8 +: 8] = strb[b] ? new_v[b*8 +: 8] : old_v[b*8 +: 8];
    end
  endfunction

  function automatic logic wr_addr_valid(input logic [7:0] a);
    case (a)
      npu_pkg::CSR_CTRL, npu_pkg::CSR_DIM_M, npu_pkg::CSR_DIM_N, npu_pkg::CSR_DIM_K, npu_pkg::CSR_A_BASE, npu_pkg::CSR_B_BASE,
      npu_pkg::CSR_C_BASE, npu_pkg::CSR_BIAS_BASE, npu_pkg::CSR_RQ_MULT, npu_pkg::CSR_RQ_SHIFT, npu_pkg::CSR_FLAGS:
        wr_addr_valid = 1'b1;
      default:
        wr_addr_valid = 1'b0;
    endcase
  endfunction

  logic [31:0] dim_m_q, dim_n_q, dim_k_q, rq_shift_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_held       <= 1'b0;
      w_held        <= 1'b0;
      aw_addr_q     <= '0;
      w_data_q      <= '0;
      w_strb_q      <= '0;
      s_axil_bvalid <= 1'b0;
      s_axil_bresp  <= RESP_OKAY;
      dim_m_q       <= '0;
      dim_n_q       <= '0;
      dim_k_q       <= '0;
      a_base        <= '0;
      b_base        <= '0;
      c_base        <= '0;
      bias_base     <= '0;
      rq_mult       <= 32'sd1;
      rq_shift_q    <= '0;
      flags_q       <= 32'h4;    // signed_act = 1
      start         <= 1'b0;
      abort         <= 1'b0;
      perf_clr      <= 1'b0;
    end else begin
      start    <= 1'b0;
      abort    <= 1'b0;
      perf_clr <= 1'b0;

      if (aw_fire && !wr_commit) begin
        aw_held   <= 1'b1;
        aw_addr_q <= s_axil_awaddr;
      end
      if (w_fire && !wr_commit) begin
        w_held   <= 1'b1;
        w_data_q <= s_axil_wdata;
        w_strb_q <= s_axil_wstrb;
      end

      if (wr_commit) begin
        aw_held       <= 1'b0;
        w_held        <= 1'b0;
        s_axil_bvalid <= 1'b1;
        s_axil_bresp  <= wr_addr_valid(wr_addr) ? RESP_OKAY : RESP_SLVERR;

        case (wr_addr)
          npu_pkg::CSR_CTRL: begin
            if (wr_strb[0]) begin
              start    <= wr_data[npu_pkg::CTRL_START_BIT] && !busy;
              abort    <= wr_data[npu_pkg::CTRL_ABORT_BIT];
              perf_clr <= wr_data[npu_pkg::CTRL_PERF_CLR_BIT];
            end
          end
          npu_pkg::CSR_DIM_M:     dim_m_q    <= apply_strb(dim_m_q,    wr_data, wr_strb);
          npu_pkg::CSR_DIM_N:     dim_n_q    <= apply_strb(dim_n_q,    wr_data, wr_strb);
          npu_pkg::CSR_DIM_K:     dim_k_q    <= apply_strb(dim_k_q,    wr_data, wr_strb);
          npu_pkg::CSR_A_BASE:    a_base     <= apply_strb(a_base,     wr_data, wr_strb);
          npu_pkg::CSR_B_BASE:    b_base     <= apply_strb(b_base,     wr_data, wr_strb);
          npu_pkg::CSR_C_BASE:    c_base     <= apply_strb(c_base,     wr_data, wr_strb);
          npu_pkg::CSR_BIAS_BASE: bias_base  <= apply_strb(bias_base,  wr_data, wr_strb);
          npu_pkg::CSR_RQ_MULT:   rq_mult    <= apply_strb(rq_mult,    wr_data, wr_strb);
          npu_pkg::CSR_RQ_SHIFT:  rq_shift_q <= apply_strb(rq_shift_q, wr_data, wr_strb);
          npu_pkg::CSR_FLAGS:     flags_q    <= apply_strb(flags_q,    wr_data, wr_strb);
          default: ;
        endcase
      end else if (s_axil_bvalid && s_axil_bready) begin
        s_axil_bvalid <= 1'b0;
      end
    end
  end

  assign dim_m      = dim_m_q[15:0];
  assign dim_n      = dim_n_q[15:0];
  assign dim_k      = dim_k_q[15:0];
  assign rq_shift   = rq_shift_q[4:0];
  assign relu_en    = flags_q[npu_pkg::FLAG_RELU_EN_BIT];
  assign bias_en    = flags_q[npu_pkg::FLAG_BIAS_EN_BIT];
  assign signed_act = flags_q[npu_pkg::FLAG_SIGNED_ACT_BIT];

  // --------------------------------------------------------------------------
  // Sticky status
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_q <= 1'b0;
      err_q  <= 1'b0;
    end else begin
      if (start) begin
        done_q <= 1'b0;
        err_q  <= 1'b0;
      end
      if (done_pulse) done_q <= 1'b1;
      if (err_pulse)  err_q  <= 1'b1;
    end
  end

  assign irq = done_q;

  // --------------------------------------------------------------------------
  // Read channel
  // --------------------------------------------------------------------------
  assign s_axil_arready = !s_axil_rvalid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_axil_rvalid <= 1'b0;
      s_axil_rdata  <= '0;
      s_axil_rresp  <= RESP_OKAY;
    end else begin
      if (s_axil_arvalid && s_axil_arready) begin
        s_axil_rvalid <= 1'b1;
        s_axil_rresp  <= RESP_OKAY;
        case (s_axil_araddr)
          npu_pkg::CSR_CTRL:        s_axil_rdata <= '0;
          npu_pkg::CSR_STATUS:      s_axil_rdata <= {29'd0, err_q, done_q, busy};
          npu_pkg::CSR_DIM_M:       s_axil_rdata <= dim_m_q;
          npu_pkg::CSR_DIM_N:       s_axil_rdata <= dim_n_q;
          npu_pkg::CSR_DIM_K:       s_axil_rdata <= dim_k_q;
          npu_pkg::CSR_A_BASE:      s_axil_rdata <= a_base;
          npu_pkg::CSR_B_BASE:      s_axil_rdata <= b_base;
          npu_pkg::CSR_C_BASE:      s_axil_rdata <= c_base;
          npu_pkg::CSR_BIAS_BASE:   s_axil_rdata <= bias_base;
          npu_pkg::CSR_RQ_MULT:     s_axil_rdata <= rq_mult;
          npu_pkg::CSR_RQ_SHIFT:    s_axil_rdata <= rq_shift_q;
          npu_pkg::CSR_FLAGS:       s_axil_rdata <= flags_q;
          npu_pkg::CSR_PERF_CYCLES: s_axil_rdata <= perf_cycles;
          npu_pkg::CSR_PERF_MACS:   s_axil_rdata <= perf_macs;
          npu_pkg::CSR_PERF_STALL:  s_axil_rdata <= perf_stall;
          npu_pkg::CSR_HW_CFG:      s_axil_rdata <= hw_cfg;
          default: begin
            s_axil_rdata <= '0;
            s_axil_rresp <= RESP_SLVERR;
          end
        endcase
      end else if (s_axil_rvalid && s_axil_rready) begin
        s_axil_rvalid <= 1'b0;
      end
    end
  end

endmodule : npu_csr

`endif // NPU_CSR_SV
