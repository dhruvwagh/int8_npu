// ============================================================================
// File: npu_drain.sv
// Description: Column-by-column accumulator readout module
//              - Drains ARR_M x ARR_N array accumulators in ARR_N clock cycles
//              - CAPTURE_ON_CLR = 0: snapshots all accumulators at drain_start
//              - CAPTURE_ON_CLR = 1: PE(i,j)'s shadow register loads its
//                accumulator in the cycle the skewed clear reaches that PE,
//                i.e. just before the PE clears for the next tile. The next
//                tile can then stream back to back; drain_start only reads
//                the shadow registers.
// Author: Dhruv Wagh
// ============================================================================

`ifndef NPU_DRAIN_SV
`define NPU_DRAIN_SV

`timescale 1ns/1ps

module npu_drain #(
  parameter int ARR_M = 16,
  parameter int ARR_N = 16,
  parameter int ACC_W = 32,
  parameter bit CAPTURE_ON_CLR = 1'b0
) (
  input  logic                                clk,
  input  logic                                rst_n,
  input  logic                                drain_start,
  input  logic signed [ARR_M*ARR_N*ACC_W-1:0] acc,
  input  logic        [ARR_M*ARR_N-1:0]       clr,       // skewed clear (CAPTURE_ON_CLR only)
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

      // Per-PE capture on the clear wavefront
      if (CAPTURE_ON_CLR) begin
        for (int m = 0; m < ARR_M; m++) begin
          for (int n = 0; n < ARR_N; n++) begin
            if (clr[m*ARR_N + n]) acc_shadow[m][n] <= acc[(m*ARR_N + n)*ACC_W +: ACC_W];
          end
        end
      end

      if (drain_start && !active) begin
        // Snapshot all accumulators
        if (!CAPTURE_ON_CLR) begin
          for (int m = 0; m < ARR_M; m++) begin
            for (int n = 0; n < ARR_N; n++) begin
              acc_shadow[m][n] <= acc[(m*ARR_N + n)*ACC_W +: ACC_W];
            end
          end
        end

        active        <= 1'b1;
        drain_valid   <= 1'b1;
        drain_col_idx <= '0;
        col_cnt       <= '0;

        // Drive first column (col 0)
        for (int m = 0; m < ARR_M; m++) begin
          drain_data[m*ACC_W +: ACC_W] <= CAPTURE_ON_CLR ? acc_shadow[m][0]
                                                         : acc[(m*ARR_N + 0)*ACC_W +: ACC_W];
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
