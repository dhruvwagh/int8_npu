// ============================================================================
// File: tb_mnist.sv
// Description: Full MNIST CNN inference on npu_top
//              The testbench plays the host/driver:
//                image -> im2col -> NPU conv1 -> maxpool -> im2col -> NPU conv2
//                      -> maxpool -> flatten -> NPU fc -> argmax
//              The NPU runs every GEMM + bias + requant + ReLU; the host does
//              only data movement (im2col, 2x2 maxpool, flatten), as in SPEC v1.
//              Each layer's output is compared byte-for-byte with the integer
//              golden model (compiler/qnet.py) for every image.
// Plusargs: +n=<images> +mem_lat=<cycles> +mem_jit=<cycles> +mem_rdy=<percent>
// Author: Dhruv Wagh
// ============================================================================

`timescale 1ns/1ps

module tb_mnist;

  import npu_pkg::*;

  `include "npu_tb_harness.svh"

  localparam string VDIR = "tb/vectors/mnist/";

  // Planar feature maps: index c*H*W + y*W + x
  logic signed [7:0] fm_in  [16*28*28];
  logic signed [7:0] fm_out [16*28*28];

  // Layer descriptors from params.txt
  logic [31:0] img_base, n_img_avail;
  logic [31:0] l_a [3], l_bias [3], l_b [3], l_c [3], l_mult [3];
  int          l_m [3], l_n [3], l_k [3], l_shift [3], l_flags [3];

  longint perf_cyc_sum [3], perf_mac_sum [3], perf_stall_sum [3];
  int     layer_errors [3];

  // --------------------------------------------------------------------------
  // Host operations
  // --------------------------------------------------------------------------
  // (C, H, W) planar in fm_in -> B panel layout at b_base (3x3, pad 1)
  task automatic host_im2col(input int C, input int H, input int W, input logic [31:0] b_base);
    int K, N, n_pad, n, ci, ky, kx, y, x, yy, xx;
    logic [7:0] v;
    K = C * 9;
    N = H * W;
    n_pad = ((N + 15) / 16) * 16;
    for (int u = 0; u < n_pad / 16; u++) begin
      for (int k = 0; k < K; k++) begin
        ci = k / 9; ky = (k % 9) / 3; kx = k % 3;
        for (int c = 0; c < 16; c++) begin
          n = u * 16 + c;
          v = 8'd0;
          if (n < N) begin
            y = n / W; x = n % W;
            yy = y + ky - 1; xx = x + kx - 1;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) v = fm_in[ci*H*W + yy*W + xx];
          end
          mem_set_byte(b_base + (u * K + k) * 16 + c, v);
        end
      end
    end
  endtask

  // Output panel layout at c_base (M x N) -> planar fm_out
  task automatic host_read_output(input int M, input int N, input logic [31:0] c_base);
    for (int m = 0; m < M; m++) begin
      for (int n = 0; n < N; n++) begin
        fm_out[m*N + n] = mem_get_byte(c_base + ((m / 16) * N + n) * 16 + (m % 16));
      end
    end
  endtask

  // fm_out (C, H, W) -> 2x2 maxpool -> fm_in (C, H/2, W/2)
  task automatic host_maxpool(input int C, input int H, input int W);
    logic signed [7:0] a, b, c, d, m1, m2;
    for (int ch = 0; ch < C; ch++) begin
      for (int y = 0; y < H / 2; y++) begin
        for (int x = 0; x < W / 2; x++) begin
          a = fm_out[ch*H*W + (2*y)*W + 2*x];
          b = fm_out[ch*H*W + (2*y)*W + 2*x + 1];
          c = fm_out[ch*H*W + (2*y+1)*W + 2*x];
          d = fm_out[ch*H*W + (2*y+1)*W + 2*x + 1];
          m1 = (a > b) ? a : b;
          m2 = (c > d) ? c : d;
          fm_in[ch*(H/2)*(W/2) + y*(W/2) + x] = (m1 > m2) ? m1 : m2;
        end
      end
    end
  endtask

  // fm_in flattened (K values) -> B panel with N = 1
  task automatic host_fc_input(input int K, input logic [31:0] b_base);
    for (int k = 0; k < K; k++) begin
      for (int c = 0; c < 16; c++) begin
        mem_set_byte(b_base + k * 16 + c, (c == 0) ? fm_in[k] : 8'd0);
      end
    end
  endtask

  // --------------------------------------------------------------------------
  // Run one NPU layer and check it against the golden bytes
  // --------------------------------------------------------------------------
  task automatic npu_layer(input int li, input int fd_exp);
    int ok, errs;
    logic [31:0] cyc, macs, stall;
    logic [7:0]  e;
    csr_wr(CSR_DIM_M, l_m[li]);
    csr_wr(CSR_DIM_N, l_n[li]);
    csr_wr(CSR_DIM_K, l_k[li]);
    csr_wr(CSR_A_BASE, l_a[li]);
    csr_wr(CSR_B_BASE, l_b[li]);
    csr_wr(CSR_C_BASE, l_c[li]);
    csr_wr(CSR_BIAS_BASE, l_bias[li]);
    csr_wr(CSR_RQ_MULT, l_mult[li]);
    csr_wr(CSR_RQ_SHIFT, l_shift[li]);
    csr_wr(CSR_FLAGS, l_flags[li]);
    csr_wr(CSR_CTRL, 32'h1);
    wait_done(ok);
    if (!ok) layer_errors[li]++;

    csr_rd(CSR_PERF_CYCLES, cyc);
    csr_rd(CSR_PERF_MACS, macs);
    csr_rd(CSR_PERF_STALL, stall);
    perf_cyc_sum[li]   += cyc;
    perf_mac_sum[li]   += macs;
    perf_stall_sum[li] += stall;

    host_read_output(l_m[li], l_n[li], l_c[li]);
    errs = 0;
    for (int i = 0; i < l_m[li] * l_n[li]; i++) begin
      if ($fscanf(fd_exp, "%h", e) != 1) $fatal(1, "expected-output file ended early");
      if (fm_out[i] !== e) errs++;
    end
    layer_errors[li] += errs;
  endtask

  // --------------------------------------------------------------------------
  // Main
  // --------------------------------------------------------------------------
  initial begin
    int fd, fd_c1, fd_c2, fd_lg, fd_lab, n_img, pred, gold_pred, label, lat, jit, rdy;
    int rtl_correct, gold_correct, pred_mismatch, img_errors;
    logic signed [7:0] best;
    logic [7:0] gp;
    real util;
    static string names [3] = '{"conv1", "conv2", "fc   "};

    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
    awaddr = 0; araddr = 0; wdata = 0; wstrb = 0;

    fd = $fopen({VDIR, "params.txt"}, "r");
    if (fd == 0) $fatal(1, "Cannot open params.txt (run scripts/run_mnist_test.py)");
    void'($fscanf(fd, "%h %h", n_img_avail, img_base));
    for (int li = 0; li < 3; li++) begin
      void'($fscanf(fd, "%h %h %h %h %d %d %d %h %d %d",
                    l_a[li], l_bias[li], l_b[li], l_c[li], l_m[li], l_n[li], l_k[li],
                    l_mult[li], l_shift[li], l_flags[li]));
      perf_cyc_sum[li] = 0; perf_mac_sum[li] = 0; perf_stall_sum[li] = 0; layer_errors[li] = 0;
    end
    $fclose(fd);
    $readmemh({VDIR, "mem.hex"}, mem);

    n_img = n_img_avail;
    if ($value$plusargs("n=%d", n_img)) if (n_img > n_img_avail) n_img = n_img_avail;
    lat = 1; jit = 0; rdy = 100;
    void'($value$plusargs("mem_lat=%d", lat));
    void'($value$plusargs("mem_jit=%d", jit));
    void'($value$plusargs("mem_rdy=%d", rdy));
    mem_lat = lat; mem_jit = jit; mem_ready_pct = rdy;

    fd_c1  = $fopen({VDIR, "exp_c1.hex"}, "r");
    fd_c2  = $fopen({VDIR, "exp_c2.hex"}, "r");
    fd_lg  = $fopen({VDIR, "exp_logits.hex"}, "r");
    fd_lab = $fopen({VDIR, "labels.txt"}, "r");

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    $display("MNIST on npu_top: %0d images, memory latency %0d+%0d, ready %0d%%", n_img, lat, jit, rdy);
    rtl_correct = 0; gold_correct = 0; pred_mismatch = 0;

    for (int img = 0; img < n_img; img++) begin
      img_errors = layer_errors[0] + layer_errors[1] + layer_errors[2];

      // conv1: raw image (1, 28, 28) -> im2col
      for (int i = 0; i < 784; i++) fm_in[i] = mem_get_byte(img_base + img * 784 + i);
      host_im2col(1, 28, 28, l_b[0]);
      npu_layer(0, fd_c1);

      // pool -> conv2
      host_maxpool(8, 28, 28);
      host_im2col(8, 14, 14, l_b[1]);
      npu_layer(1, fd_c2);

      // pool -> flatten -> fc
      host_maxpool(16, 14, 14);
      host_fc_input(784, l_b[2]);
      npu_layer(2, fd_lg);

      // argmax (first maximum wins, same as numpy)
      pred = 0; best = fm_out[0];
      for (int i = 1; i < 10; i++) if (fm_out[i] > best) begin best = fm_out[i]; pred = i; end
      void'($fscanf(fd_lab, "%d %d", label, gold_pred));
      rtl_correct  += (pred == label);
      gold_correct += (gold_pred == label);
      pred_mismatch += (pred != gold_pred);

      img_errors = layer_errors[0] + layer_errors[1] + layer_errors[2] - img_errors;
      if (img_errors != 0) begin
        $display("  image %4d: label %0d  npu %0d  golden %0d  %0d BYTE MISMATCHES",
                 img, label, pred, gold_pred, img_errors);
      end else if (img < 3 || (img + 1) % 50 == 0) begin
        $display("  image %4d: label %0d  npu %0d  golden %0d  bit-exact", img, label, pred, gold_pred);
      end
    end

    $display("\nPer-layer performance (average over %0d images):", n_img);
    $display("  layer   cycles   stall   MACs      utilization");
    for (int li = 0; li < 3; li++) begin
      util = 100.0 * real'(perf_mac_sum[li]) / (real'(perf_cyc_sum[li]) * ARR_M * ARR_N);
      $display("  %s  %7.0f %7.0f  %8.0f   %5.1f%%", names[li],
               real'(perf_cyc_sum[li]) / n_img, real'(perf_stall_sum[li]) / n_img,
               real'(perf_mac_sum[li]) / n_img, util);
    end
    util = 100.0 * real'(perf_mac_sum[0] + perf_mac_sum[1] + perf_mac_sum[2]) /
           (real'(perf_cyc_sum[0] + perf_cyc_sum[1] + perf_cyc_sum[2]) * ARR_M * ARR_N);
    $display("  total   %7.0f cycles/image, %5.1f%% utilization",
             real'(perf_cyc_sum[0] + perf_cyc_sum[1] + perf_cyc_sum[2]) / n_img, util);

    $display("\nAccuracy on %0d images: NPU %0.2f%%, golden int8 model %0.2f%%",
             n_img, 100.0 * rtl_correct / n_img, 100.0 * gold_correct / n_img);
    $display("\n==================================================================");
    if (layer_errors[0] + layer_errors[1] + layer_errors[2] == 0 && pred_mismatch == 0 && total_errors == 0) begin
      $display("  MNIST PASSED: all %0d images bit-exact in every layer", n_img);
      $display("==================================================================");
    end else begin
      $display("  MNIST FAILED: byte mismatches conv1=%0d conv2=%0d fc=%0d, prediction mismatches=%0d",
               layer_errors[0], layer_errors[1], layer_errors[2], pred_mismatch);
      $display("==================================================================");
      $fatal(1, "tb_mnist failed");
    end
    $finish;
  end

endmodule : tb_mnist
