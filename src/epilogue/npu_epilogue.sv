// ============================================================================
// File: npu_epilogue.sv
// Description: Multi-Lane Parallel Requantization Epilogue Unit
//              - Processes ARR_M parallel 32-bit accumulator lanes simultaneously
//              - In 16 cycles, converts a full 16x16 tile into sixteen 128-bit memory words
//              - Pipeline operations per lane:
//                1. 32-bit Accumulator + 32-bit Bias
//                2. Signed 33x32 -> 65-bit Requantize Multiplication (RQ_MULT)
//                3. Round Half Away From Zero Right Shift (RQ_SHIFT)
//                4. ReLU Activation: max(0, x)
//                5. Saturation Clamping: [-128, 127] for signed INT8
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

  // Unpacked internal signals for each parallel lane
  logic signed [DATA_W-1:0] lane_out [ARR_M];

  generate
    for (genvar m = 0; m < ARR_M; m++) begin : gen_lanes
      wire signed [ACC_W-1:0] lane_acc  = in_acc[m*ACC_W +: ACC_W];
      wire signed [ACC_W-1:0] lane_bias = in_bias[m*ACC_W +: ACC_W];

      logic signed [32:0] biased;
      logic signed [64:0] prod;
      logic signed [63:0] shifted;
      logic signed [63:0] relu_val;
      logic signed [7:0]  sat_val;
      logic [64:0]        abs_p;
      logic [63:0]        abs_rounded;

      always_comb begin
        biased      = '0;
        prod        = '0;
        shifted     = '0;
        relu_val    = '0;
        sat_val     = '0;
        abs_p       = '0;
        abs_rounded = '0;

        // 1. Bias Addition
        if (bias_en) begin
          biased = 33'(lane_acc) + 33'(lane_bias);
        end else begin
          biased = 33'(lane_acc);
        end

        // 2. Signed Multiplier: 33b x 32b -> 65b
        prod = 65'(biased) * 65'($signed(rq_mult));

        // 3. Round Half Away From Zero Right Shift
        if (rq_shift > 0) begin
          if (prod >= 0) begin
            shifted = 64'((prod + (65'd1 << (rq_shift - 1))) >>> rq_shift);
          end else begin
            abs_p       = -prod;
            abs_rounded = 64'((abs_p + (65'd1 << (rq_shift - 1))) >> rq_shift);
            shifted     = 64'd0 - abs_rounded;
          end
        end else begin
          shifted = 64'(prod);
        end

        // 4. ReLU
        if (relu_en && (shifted < 0)) begin
          relu_val = '0;
        end else begin
          relu_val = shifted;
        end

        // 5. Saturation / Clamping
        if (signed_act) begin
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

  // --------------------------------------------------------------------------
  // Output Pipeline Register (1-cycle synchronous latency)
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid   <= 1'b0;
      out_col_idx <= '0;
      out_data    <= '0;
    end else begin
      out_valid   <= in_valid;
      out_col_idx <= in_col_idx;
      for (int m = 0; m < ARR_M; m++) begin
        out_data[m*DATA_W +: DATA_W] <= lane_out[m];
      end
    end
  end

endmodule : npu_epilogue

`endif // NPU_EPILOGUE_SV
