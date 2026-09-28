// ============================================================================
// File: npu_drain.sv
// Description: Column-by-column accumulator readout module
//              - Drains ARR_M x ARR_N array accumulators in ARR_N clock cycles
//              - Captures accumulator snapshot at drain_start to allow
//                overlapping the drain with the next tile's fill
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_DRAIN_SV
`define NPU_DRAIN_SV

`timescale 1ns/1ps

module npu_drain #(
  parameter int ARR_M = 16,
  parameter int ARR_N = 16,
  parameter int ACC_W = 32
) (
  input  logic                                clk,
  input  logic                                rst_n,
  input  logic                                drain_start,
  input  logic signed [ARR_M*ARR_N*ACC_W-1:0] acc,
  output logic                                drain_busy,
  output logic                                drain_valid,
  output logic [$clog2(ARR_N)-1:0]            drain_col_idx,
  output logic signed [ARR_M*ACC_W-1:0]       drain_data,
  output logic                                drain_done
);

  localparam int COL_W = (ARR_N > 1) ? $clog2(ARR_N) : 1;

  // Shadow registers to hold tile outputs during drain (enables fill/drain overlap)
  logic signed [ACC_W-1:0] acc_shadow [ARR_M][ARR_N];
  logic [COL_W-1:0]        col_cnt;
  logic                    active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      col_cnt       <= '0;
      active        <= 1'b0;
      drain_done    <= 1'b0;
      drain_valid   <= 1'b0;
      drain_col_idx <= '0;
      drain_data    <= '0;
      for (int m = 0; m < ARR_M; m++) begin
        for (int n = 0; n < ARR_N; n++) begin
          acc_shadow[m][n] <= '0;
        end
      end
    end else begin
      drain_done <= 1'b0;

      if (drain_start && !active) begin
        // Snapshot all accumulators
        for (int m = 0; m < ARR_M; m++) begin
          for (int n = 0; n < ARR_N; n++) begin
            acc_shadow[m][n] <= acc[(m*ARR_N + n)*ACC_W +: ACC_W];
          end
        end

        active        <= 1'b1;
        drain_valid   <= 1'b1;
        drain_col_idx <= '0;
        col_cnt       <= '0;

        // Drive first column (col 0)
        for (int m = 0; m < ARR_M; m++) begin
          drain_data[m*ACC_W +: ACC_W] <= acc[(m*ARR_N + 0)*ACC_W +: ACC_W];
        end

        if (ARR_N == 1) begin
          drain_done <= 1'b1;
          active     <= 1'b0;
        end
      end else if (active) begin
        if (col_cnt == COL_W'(ARR_N - 1)) begin
          active      <= 1'b0;
          drain_valid <= 1'b0;
          drain_done  <= 1'b1;
        end else begin
          col_cnt       <= col_cnt + 1'b1;
          drain_col_idx <= col_cnt + 1'b1;
          for (int m = 0; m < ARR_M; m++) begin
            drain_data[m*ACC_W +: ACC_W] <= acc_shadow[m][col_cnt + 1'b1];
          end
        end
      end else begin
        drain_valid <= 1'b0;
      end
    end
  end

  assign drain_busy = active | drain_start;

endmodule : npu_drain

`endif // NPU_DRAIN_SV
