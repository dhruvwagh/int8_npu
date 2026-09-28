// ============================================================================
// File: npu_array.sv
// Description: Parameterized ARR_M x ARR_N 2D Systolic Array of PEs
//              - True systolic topology: operands hop PE-to-PE with registered delays
//              - Output-stationary dataflow: accumulators accumulate in-place
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_ARRAY_SV
`define NPU_ARRAY_SV

`timescale 1ns/1ps

module npu_array #(
  parameter int ARR_M = 16,
  parameter int ARR_N = 16,
  parameter int ACC_W = 32
) (
  input  logic                                clk,
  input  logic                                rst_n,
  input  logic                                en,                     // Advance PE pipelines
  input  logic        [ARR_M*ARR_N-1:0]       clr,                    // Systolic clear per PE
  input  logic signed [ARR_M*8-1:0]           a_in,                   // West inputs (activations)
  input  logic signed [ARR_N*8-1:0]           b_in,                   // North inputs (weights)
  output logic signed [ARR_M*8-1:0]           a_out,                  // East outputs
  output logic signed [ARR_N*8-1:0]           b_out,                  // South outputs
  output logic signed [ARR_M*ARR_N*ACC_W-1:0] acc                     // Stationed accumulator matrix
);

  // --------------------------------------------------------------------------
  // Inter-PE Interconnect Wires
  // --------------------------------------------------------------------------
  // Horizontal interconnect: a_h[i][j] enters PE(i, j) from the west
  logic signed [7:0] a_h [ARR_M][ARR_N+1];

  // Vertical interconnect: b_v[i][j] enters PE(i, j) from the north
  logic signed [7:0] b_v [ARR_M+1][ARR_N];

  // Boundary connections: West and North edges
  generate
    for (genvar i = 0; i < ARR_M; i++) begin : gen_west_boundary
      assign a_h[i][0]              = a_in[i*8 +: 8];
      assign a_out[i*8 +: 8]        = a_h[i][ARR_N];
    end

    for (genvar j = 0; j < ARR_N; j++) begin : gen_north_boundary
      assign b_v[0][j]              = b_in[j*8 +: 8];
      assign b_out[j*8 +: 8]        = b_v[ARR_M][j];
    end
  endgenerate

  // --------------------------------------------------------------------------
  // 2D Array PE Instantiation
  // --------------------------------------------------------------------------
  generate
    for (genvar i = 0; i < ARR_M; i++) begin : gen_row
      for (genvar j = 0; j < ARR_N; j++) begin : gen_col
        wire clr_pe = clr[i*ARR_N + j];
        wire signed [ACC_W-1:0] acc_pe;

        npu_pe #(
          .ACC_W(ACC_W)
        ) pe_inst (
          .clk   (clk),
          .rst_n (rst_n),
          .en    (en),
          .clr   (clr_pe),
          .a_in  (a_h[i][j]),
          .b_in  (b_v[i][j]),
          .a_out (a_h[i][j+1]),
          .b_out (b_v[i+1][j]),
          .acc   (acc_pe)
        );

        assign acc[(i*ARR_N + j)*ACC_W +: ACC_W] = acc_pe;
      end
    end
  endgenerate

endmodule : npu_array

`endif // NPU_ARRAY_SV
