// ============================================================================
// File: npu_skew.sv
// Description: Triangular skew delay registers for systolic array inputs
//              - Row i activations delayed by i cycles
//              - Col j weights delayed by j cycles
//              - Systolic tile clear delayed by (i + j) cycles for PE(i, j)
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_SKEW_SV
`define NPU_SKEW_SV

`timescale 1ns/1ps

module npu_skew #(
  parameter int ARR_M  = 16,
  parameter int ARR_N  = 16,
  parameter int DATA_W = 8
) (
  input  logic                                clk,
  input  logic                                rst_n,
  input  logic                                en,         // Clock enable
  input  logic                                clr_in,     // Clear pulse at tile cycle k=0
  input  logic signed [ARR_M*DATA_W-1:0]      a_in,
  input  logic signed [ARR_N*DATA_W-1:0]      b_in,
  output logic signed [ARR_M*DATA_W-1:0]      a_skewed,
  output logic signed [ARR_N*DATA_W-1:0]      b_skewed,
  output logic        [ARR_M*ARR_N-1:0]       clr_skewed
);

  // --------------------------------------------------------------------------
  // Activation Input Skew (Row i delayed by i cycles)
  // --------------------------------------------------------------------------
  generate
    for (genvar i = 0; i < ARR_M; i++) begin : gen_a_skew
      wire signed [DATA_W-1:0] a_row_in = a_in[i*DATA_W +: DATA_W];
      if (i == 0) begin : gen_a_direct
        assign a_skewed[0*DATA_W +: DATA_W] = a_row_in;
      end else begin : gen_a_delay
        logic signed [DATA_W-1:0] shift_reg [i];
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            for (int s = 0; s < i; s++) begin
              shift_reg[s] <= '0;
            end
          end else if (en) begin
            shift_reg[0] <= a_row_in;
            for (int s = 1; s < i; s++) begin
              shift_reg[s] <= shift_reg[s-1];
            end
          end
        end
        assign a_skewed[i*DATA_W +: DATA_W] = shift_reg[i-1];
      end
    end
  endgenerate

  // --------------------------------------------------------------------------
  // Weight Input Skew (Column j delayed by j cycles)
  // --------------------------------------------------------------------------
  generate
    for (genvar j = 0; j < ARR_N; j++) begin : gen_b_skew
      wire signed [DATA_W-1:0] b_col_in = b_in[j*DATA_W +: DATA_W];
      if (j == 0) begin : gen_b_direct
        assign b_skewed[0*DATA_W +: DATA_W] = b_col_in;
      end else begin : gen_b_delay
        logic signed [DATA_W-1:0] shift_reg [j];
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            for (int s = 0; s < j; s++) begin
              shift_reg[s] <= '0;
            end
          end else if (en) begin
            shift_reg[0] <= b_col_in;
            for (int s = 1; s < j; s++) begin
              shift_reg[s] <= shift_reg[s-1];
            end
          end
        end
        assign b_skewed[j*DATA_W +: DATA_W] = shift_reg[j-1];
      end
    end
  endgenerate

  // --------------------------------------------------------------------------
  // Systolic Clear Skew (PE(i, j) receives clear at cycle i + j)
  // Uses a single unified shift register of length (ARR_M + ARR_N - 2)
  // --------------------------------------------------------------------------
  localparam int MAX_DELAY = ARR_M + ARR_N - 2;
  logic [MAX_DELAY:1] clr_pipe;

  generate
    if (MAX_DELAY > 0) begin : gen_clr_pipe
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          clr_pipe <= '0;
        end else if (en) begin
          clr_pipe[1] <= clr_in;
          for (int d = 2; d <= MAX_DELAY; d++) begin
            clr_pipe[d] <= clr_pipe[d-1];
          end
        end
      end
    end
  endgenerate

  // Route clear signal to PE(i, j)
  generate
    for (genvar i = 0; i < ARR_M; i++) begin : gen_clr_row
      for (genvar j = 0; j < ARR_N; j++) begin : gen_clr_col
        localparam int D = i + j;
        if (D == 0) begin : gen_clr_d0
          assign clr_skewed[i*ARR_N + j] = clr_in;
        end else begin : gen_clr_dn
          assign clr_skewed[i*ARR_N + j] = clr_pipe[D];
        end
      end
    end
  endgenerate

endmodule : npu_skew

`endif // NPU_SKEW_SV
