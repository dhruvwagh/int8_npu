// ============================================================================
// File: npu_epilogue.sv
// Description: Multi-Lane Parallel Requantization Epilogue Unit
//              - Processes ARR_M parallel 32-bit accumulator lanes simultaneously
//              - In 16 cycles, converts a full 16x16 tile into sixteen 128-bit memory words
//              - 2-stage pipeline per lane (LATENCY = 2 cycles, 1 column/cycle):
//                Stage 1: 32-bit Accumulator + 32-bit Bias
//                         Signed 33x32 -> 65-bit Requantize Multiplication (RQ_MULT)
//                Stage 2: Round Half Away From Zero Right Shift (RQ_SHIFT)
//                         ReLU Activation: max(0, x)
//                         Saturation Clamping: [-128, 127] signed / [0, 255] unsigned
//              - Bias is consumed in stage 1, so in_bias may change once the last
//                column has been accepted.
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_EPILOGUE_SV
`define NPU_EPILOGUE_SV

`timescale 1ns/1ps

module npu_epilogue #(
  parameter int ARR_M  = 16,
  parameter int ARR_N  = 16,
  parameter int ACC_W  = 32,
  parameter int DATA_W = 8
) (
  input  logic                                clk,
  input  logic                                rst_n,

  // Runtime Requantization Configuration
  input  logic                                bias_en,
  input  logic signed [31:0]                  rq_mult,
  input  logic [4:0]                          rq_shift,
  input  logic                                relu_en,
  input  logic                                signed_act,

  // Column Stream Input (from npu_drain)
  input  logic                                in_valid,
  input  logic [$clog2(ARR_N)-1:0]            in_col_idx,
  input  logic signed [ARR_M*ACC_W-1:0]       in_acc,
  input  logic signed [ARR_M*ACC_W-1:0]       in_bias,

  // Quantized Stream Output (to npu_obuf / tensor memory)
  output logic                                out_valid,
  output logic [$clog2(ARR_N)-1:0]            out_col_idx,
  output logic signed [ARR_M*DATA_W-1:0]      out_data
);

  localparam int COL_W = (ARR_N > 1) ? $clog2(ARR_N) : 1;

  // --------------------------------------------------------------------------
  // Stage 1: bias add + requantize multiply
  // --------------------------------------------------------------------------
  logic signed [64:0] prod_s1 [ARR_M];

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_s1
      wire signed [ACC_W-1:0] lane_acc  = in_acc[m*ACC_W +: ACC_W];
      wire signed [ACC_W-1:0] lane_bias = in_bias[m*ACC_W +: ACC_W];
      logic signed [32:0] biased;
      logic signed [64:0] prod;

      always_comb begin
        biased = bias_en ? (33'(lane_acc) + 33'(lane_bias)) : 33'(lane_acc);
        prod   = 65'(biased) * 65'($signed(rq_mult));
      end

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          prod_s1[m] <= '0;
        end else if (in_valid) begin
          prod_s1[m] <= prod;
        end
      end
    end
  endgenerate

  logic             valid_s1;
  logic [COL_W-1:0] col_s1;
  logic [4:0]       shift_s1;
  logic             relu_s1;
  logic             signed_s1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_s1  <= 1'b0;
      col_s1    <= '0;
      shift_s1  <= '0;
      relu_s1   <= 1'b0;
      signed_s1 <= 1'b1;
    end else begin
      valid_s1 <= in_valid;
      if (in_valid) begin
        col_s1    <= in_col_idx;
        shift_s1  <= rq_shift;
        relu_s1   <= relu_en;
        signed_s1 <= signed_act;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Stage 2: rounding shift + ReLU + saturation
  // --------------------------------------------------------------------------
  logic signed [DATA_W-1:0] lane_out [ARR_M];

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_s2
      logic signed [64:0] prod;
      logic signed [63:0] shifted;
      logic signed [63:0] relu_val;
      logic signed [7:0]  sat_val;
      logic [64:0]        abs_p;
      logic [63:0]        abs_rounded;

      always_comb begin
        prod        = prod_s1[m];
        shifted     = '0;
        abs_p       = '0;
        abs_rounded = '0;

        // Round half away from zero: round |prod|, then restore the sign
        if (shift_s1 > 0) begin
          if (prod >= 0) begin
            shifted = 64'((prod + (65'd1 << (shift_s1 - 1))) >>> shift_s1);
          end else begin
            abs_p       = -prod;
            abs_rounded = 64'((abs_p + (65'd1 << (shift_s1 - 1))) >> shift_s1);
            shifted     = 64'd0 - abs_rounded;
          end
        end else begin
          shifted = 64'(prod);
        end

        // ReLU
        relu_val = (relu_s1 && (shifted < 0)) ? '0 : shifted;

        // Saturation / Clamping
        if (signed_s1) begin
          if (relu_val > 64'sd127) begin
            sat_val = 8'sd127;
          end else if (relu_val < -64'sd128) begin
            sat_val = -8'sd128;
          end else begin
            sat_val = 8'(relu_val);
          end
        end else begin
          if (relu_val > 64'sd255) begin
            sat_val = 8'hFF;
          end else if (relu_val < 64'sd0) begin
            sat_val = 8'h00;
          end else begin
            sat_val = 8'(relu_val);
          end
        end

        lane_out[m] = sat_val;
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid   <= 1'b0;
      out_col_idx <= '0;
      out_data    <= '0;
    end else begin
      out_valid <= valid_s1;
      if (valid_s1) begin
        out_col_idx <= col_s1;
        for (int m = 0; m < ARR_M; m++) begin
          out_data[m*DATA_W +: DATA_W] <= lane_out[m];
        end
      end
    end
  end

endmodule : npu_epilogue

`endif // NPU_EPILOGUE_SV
