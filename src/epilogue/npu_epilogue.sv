// ============================================================================
// File: npu_epilogue.sv
// Description: Multi-Lane Parallel Requantization Epilogue Unit
//              - Processes ARR_M parallel 32-bit accumulator lanes simultaneously
//              - In 16 cycles, converts a full 16x16 tile into sixteen 128-bit memory words
//              - 4-stage pipeline per lane (LATENCY = 4 cycles, 1 column/cycle):
//                Stage 0:  32-bit Accumulator + 32-bit Bias
//                Stage 1:  Signed 33x32 -> 65-bit Requantize Multiplication (RQ_MULT)
//                Stage 2a: Round Half Away From Zero Right Shift (RQ_SHIFT) on |x|
//                Stage 2b: sign, ReLU, saturation [-128, 127] / [0, 255]
//              - Bias is consumed in stage 0, so in_bias may change once the last
//                column has been accepted.
//              - rq_mult, rq_shift, relu_en, signed_act and bias_en are layer
//                configuration, static while the NPU is busy (write-protected in
//                npu_csr), so they are used directly and not pipelined.
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
  // Stage 0: bias add
  // --------------------------------------------------------------------------
  logic signed [32:0] biased_s0 [ARR_M];
  logic               valid_s0;
  logic [COL_W-1:0]   col_s0;

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_s0
      wire signed [ACC_W-1:0] lane_acc  = in_acc[m*ACC_W +: ACC_W];
      wire signed [ACC_W-1:0] lane_bias = in_bias[m*ACC_W +: ACC_W];
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          biased_s0[m] <= '0;
        end else if (in_valid) begin
          biased_s0[m] <= bias_en ? (33'(lane_acc) + 33'(lane_bias)) : 33'(lane_acc);
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_s0 <= 1'b0;
      col_s0   <= '0;
    end else begin
      valid_s0 <= in_valid;
      if (in_valid) col_s0 <= in_col_idx;
    end
  end

  // --------------------------------------------------------------------------
  // Stage 1: requantize multiply (the widest logic in the design, so it gets
  // a stage of its own)
  // --------------------------------------------------------------------------
  logic signed [64:0] prod_s1 [ARR_M];

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_s1
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          prod_s1[m] <= '0;
        end else if (valid_s0) begin
          prod_s1[m] <= 65'(biased_s0[m]) * 65'($signed(rq_mult));
        end
      end
    end
  endgenerate

  logic             valid_s1;
  logic [COL_W-1:0] col_s1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_s1 <= 1'b0;
      col_s1   <= '0;
    end else begin
      valid_s1 <= valid_s0;
      if (valid_s0) col_s1 <= col_s0;
    end
  end

  // --------------------------------------------------------------------------
  // Stage 2a: rounding shift on the magnitude
  //   round_half_away(p / 2^s) = sign(p) * ((|p| >> s) + |p|[s-1])
  // which equals ((|p| + 2^(s-1)) >> s) but needs no 65-bit add before the
  // shift: only a 1-bit increment after it. Any magnitude >= 256 saturates in
  // every output mode, so only a clamped 9-bit magnitude and the sign are kept.
  // --------------------------------------------------------------------------
  logic [8:0]       mag_s2 [ARR_M];
  logic             neg_s2 [ARR_M];
  logic             valid_s2;
  logic [COL_W-1:0] col_s2;

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_s2
      logic [64:0] abs_p;
      logic [64:0] q;
      logic        round_bit;
      logic [64:0] mag;

      always_comb begin
        abs_p     = prod_s1[m][64] ? (65'd0 - 65'(prod_s1[m])) : 65'(prod_s1[m]);
        q         = abs_p >> rq_shift;
        round_bit = (rq_shift != 0) ? abs_p[7'(rq_shift) - 7'd1] : 1'b0;
        mag       = q + 65'(round_bit);
      end

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          mag_s2[m] <= '0;
          neg_s2[m] <= 1'b0;
        end else if (valid_s1) begin
          mag_s2[m] <= (mag > 65'd256) ? 9'd256 : 9'(mag);
          neg_s2[m] <= prod_s1[m][64];
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_s2 <= 1'b0;
      col_s2   <= '0;
    end else begin
      valid_s2 <= valid_s1;
      if (valid_s1) col_s2 <= col_s1;
    end
  end

  // --------------------------------------------------------------------------
  // Stage 2b: sign, ReLU, saturation (all on a 9-bit magnitude)
  // --------------------------------------------------------------------------
  logic signed [DATA_W-1:0] lane_out [ARR_M];

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_s3
      always_comb begin
        if (neg_s2[m] && mag_s2[m] != 0) begin
          // Negative result
          if (relu_en || !signed_act) begin
            lane_out[m] = 8'sd0;                               // ReLU / unsigned floor
          end else if (mag_s2[m] >= 9'd128) begin
            lane_out[m] = -8'sd128;
          end else begin
            lane_out[m] = 8'(9'd0 - mag_s2[m]);
          end
        end else begin
          // Zero or positive result
          if (signed_act) begin
            lane_out[m] = (mag_s2[m] > 9'd127) ? 8'sd127 : 8'(mag_s2[m]);
          end else begin
            lane_out[m] = (mag_s2[m] > 9'd255) ? 8'hFF : 8'(mag_s2[m]);
          end
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid   <= 1'b0;
      out_col_idx <= '0;
      out_data    <= '0;
    end else begin
      out_valid <= valid_s2;
      if (valid_s2) begin
        out_col_idx <= col_s2;
        for (int m = 0; m < ARR_M; m++) begin
          out_data[m*DATA_W +: DATA_W] <= lane_out[m];
        end
      end
    end
  end

endmodule : npu_epilogue

`endif // NPU_EPILOGUE_SV
