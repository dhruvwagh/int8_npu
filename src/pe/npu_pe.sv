// ============================================================================
// File: npu_pe.sv
// Description: Output-stationary Processing Element (PE) for INT8 Systolic Array
//              - Signed 8x8 multiply with ACC_W accumulator
//              - Systolic forwarding registers (a_out, b_out)
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_PE_SV
`define NPU_PE_SV

`timescale 1ns/1ps

module npu_pe #(
  parameter int ACC_W = 32
) (
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic                     en,        // Advance computation and registers
  input  logic                     clr,       // Clear accumulator (new output tile)
  input  logic signed [7:0]        a_in,      // Activation input from west
  input  logic signed [7:0]        b_in,      // Weight input from north
  output logic signed [7:0]        a_out,     // Activation forwarded to east
  output logic signed [7:0]        b_out,     // Weight forwarded to south
  output logic signed [ACC_W-1:0]  acc        // Output-stationary accumulator value
);

  // Signed 8x8 multiplication -> signed 16-bit product
  logic signed [15:0] prod;
  assign prod = a_in * b_in;

  // Sign-extended product to accumulator width
  logic signed [ACC_W-1:0] prod_ext;
  assign prod_ext = {{ (ACC_W-16){prod[15]} }, prod};

  // Sequential register updates
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out <= 8'sd0;
      b_out <= 8'sd0;
      acc   <= '0;
    end else begin
      // Systolic forwarding registers advance only when enabled
      if (en) begin
        a_out <= a_in;
        b_out <= b_in;
      end

      // Accumulator update logic
      if (clr) begin
        if (en) begin
          // Start of new reduction tile with active operand pair
          acc <= prod_ext;
        end else begin
          // Synchronous clear while stalled / idle
          acc <= '0;
        end
      end else if (en) begin
        // Normal accumulation step
        acc <= acc + prod_ext;
      end
    end
  end

endmodule : npu_pe

`endif // NPU_PE_SV
