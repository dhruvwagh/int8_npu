// ============================================================================
// File: npu_fpga_top.sv
// Description: FPGA wrapper for npu_top
//              - On-chip tensor memory in block RAM (MEM_DW/32 lanes of 32
//                bits), so the ARR*8-bit memory port stays on chip and the
//                design fits the package I/O
//              - Host port: 32-bit word access to tensor memory for loading
//                inputs and reading results. While host_en is high the NPU's
//                memory requests are held off (mem_ready = 0).
//              - AXI4-Lite CSR port passed through
//              Memory timing seen by the NPU: always ready (unless the host is
//              accessing), read data one cycle after the request.
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module npu_fpga_top #(
  parameter int ARR    = 16,              // ARR_M = ARR_N
  parameter int MEM_KB = 64               // on-chip tensor memory
) (
  input  logic        clk,
  input  logic        rst_n,

  // AXI4-Lite CSR port
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [7:0]  s_axil_awaddr,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  input  logic [31:0] s_axil_wdata,
  input  logic [3:0]  s_axil_wstrb,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  output logic [1:0]  s_axil_bresp,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  input  logic [7:0]  s_axil_araddr,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,
  output logic [31:0] s_axil_rdata,
  output logic [1:0]  s_axil_rresp,

  // Host access to tensor memory (byte address of a 32-bit word)
  input  logic        host_en,
  input  logic        host_we,
  input  logic [15:0] host_addr,
  input  logic [31:0] host_wdata,
  output logic [31:0] host_rdata,

  output logic        irq
);

  localparam int MEM_DW = ARR * 8;        // one panel word per beat
  localparam int WB     = MEM_DW / 8;     // bytes per word
  localparam int DEPTH  = MEM_KB * 1024 / WB;
  localparam int AW     = $clog2(DEPTH);
  localparam int LANES  = MEM_DW / 32;
  localparam int LW     = (LANES > 1) ? $clog2(LANES) : 1;
  localparam int BS     = $clog2(WB);     // byte-address shift of a word

  logic              mem_req, mem_we, mem_ready, mem_rvalid;
  logic [31:0]       mem_addr;
  logic [MEM_DW-1:0] mem_wdata, mem_rdata;

  npu_top #(.ARR_M(ARR), .ARR_N(ARR), .MEM_DW(MEM_DW)) u_npu (
    .clk, .rst_n,
    .s_axil_awvalid, .s_axil_awready, .s_axil_awaddr,
    .s_axil_wvalid, .s_axil_wready, .s_axil_wdata, .s_axil_wstrb,
    .s_axil_bvalid, .s_axil_bready, .s_axil_bresp,
    .s_axil_arvalid, .s_axil_arready, .s_axil_araddr,
    .s_axil_rvalid, .s_axil_rready, .s_axil_rdata, .s_axil_rresp,
    .mem_req, .mem_we, .mem_addr, .mem_wdata, .mem_ready, .mem_rvalid, .mem_rdata,
    .irq
  );

  // --------------------------------------------------------------------------
  // Tensor memory: one address for all lanes; the host writes a single lane
  // --------------------------------------------------------------------------
  assign mem_ready = !host_en;

  wire [AW-1:0] ram_addr = host_en ? host_addr[AW+BS-1:BS] : mem_addr[AW+BS-1:BS];
  wire          npu_rd   = mem_req && mem_ready && !mem_we;
  wire          npu_wr   = mem_req && mem_ready &&  mem_we;

  logic [31:0] lane_rdata [LANES];

  generate
    for (genvar l = 0; l < LANES; l++) begin : g_lane
      logic [31:0] ram [DEPTH];
      wire         we    = host_en ? (host_we && host_addr[BS-1:2] == LW'(l)) : npu_wr;
      wire  [31:0] wdata = host_en ? host_wdata : mem_wdata[l*32 +: 32];
      always_ff @(posedge clk) begin
        if (we) ram[ram_addr] <= wdata;
        lane_rdata[l] <= ram[ram_addr];
      end
      assign mem_rdata[l*32 +: 32] = lane_rdata[l];
    end
  endgenerate

  logic [LW-1:0] host_lane_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mem_rvalid  <= 1'b0;
      host_lane_q <= '0;
    end else begin
      mem_rvalid  <= npu_rd;
      host_lane_q <= host_addr[BS-1:2];
    end
  end
  assign host_rdata = lane_rdata[host_lane_q];

  logic _unused;
  assign _unused = &{1'b0, mem_addr[31:AW+BS], mem_addr[BS-1:0], host_addr[1:0], 1'b0};

endmodule : npu_fpga_top
