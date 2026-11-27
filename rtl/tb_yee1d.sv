// Self-checking testbench of rtl/yee1d.sv (docs/rtl.md).
//
// Reads the golden vectors of src/bin/yee1d_vectors.rs from +VEC=<dir>
// (default rtl/vectors/nominal), runs T steps, and compares every E and H
// word and the sticky overflow flag after every step, bit for bit. It prints
// one summary line and ends with $fatal if any word differs.
//
// Example (Verilator 5):
//   $ verilator --binary -j 0 --top-module tb_yee1d rtl/yee1d.sv rtl/tb_yee1d.sv
//   $ obj_dir/Vtb_yee1d +VEC=rtl/vectors/nominal
//   $ obj_dir/Vtb_yee1d +VEC=rtl/vectors/saturate
`timescale 1ns / 1ps

module tb_yee1d;
    localparam int N = 64;
    localparam int T = 400;
    localparam int W = 18;

    logic clk;
    initial clk = 1'b0;
    logic rst, start, done, ovf;
    logic signed [W-1:0] src, ch;
    logic signed [W-1:0] cb [N];
    logic signed [W-1:0] e [N];
    logic signed [W-1:0] h [N];

    logic [W-1:0] cb_mem  [N];
    logic [W-1:0] ch_mem  [1];
    logic [W-1:0] src_mem [T];
    logic [W-1:0] e_mem   [T*N];
    logic [W-1:0] h_mem   [T*N];
    logic [3:0]   ovf_mem [T];

    yee1d #(.N(N), .W(W), .CW(W), .CF(16), .SRC(8)) dut (
        .clk, .rst, .start, .src, .ch, .cb, .e, .h, .done, .ovf
    );

    always #5 clk = ~clk;

    string dir;
    int errors = 0;
    int checked = 0;
    int first_bad = -1;

    initial begin
        if (!$value$plusargs("VEC=%s", dir)) dir = "rtl/vectors/nominal";
        $readmemh({dir, "/cb.hex"}, cb_mem);
        $readmemh({dir, "/ch.hex"}, ch_mem);
        $readmemh({dir, "/src.hex"}, src_mem);
        $readmemh({dir, "/e.hex"}, e_mem);
        $readmemh({dir, "/h.hex"}, h_mem);
        $readmemh({dir, "/ovf.hex"}, ovf_mem);
        for (int i = 0; i < N; i++) cb[i] = cb_mem[i];
        ch = ch_mem[0];
        rst = 1'b1;
        start = 1'b0;
        src = '0;
        repeat (2) @(posedge clk);
        rst = 1'b0;
        for (int k = 0; k < T; k++) begin
            @(negedge clk);
            src = src_mem[k];
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (done === 1'b1);
            @(negedge clk);
            for (int i = 0; i < N; i++) begin
                checked += 2;
                if (e[i] !== e_mem[k*N + i]) begin
                    errors++;
                    if (first_bad < 0) begin
                        first_bad = k;
                        $display("first mismatch: step %0d E[%0d] = %05h, expected %05h", k, i, e[i] & 18'h3ffff, e_mem[k*N + i]);
                    end
                end
                if (h[i] !== h_mem[k*N + i]) begin
                    errors++;
                    if (first_bad < 0) begin
                        first_bad = k;
                        $display("first mismatch: step %0d H[%0d] = %05h, expected %05h", k, i, h[i] & 18'h3ffff, h_mem[k*N + i]);
                    end
                end
            end
            checked++;
            if (ovf !== ovf_mem[k][0]) begin
                errors++;
                if (first_bad < 0) begin
                    first_bad = k;
                    $display("first mismatch: step %0d ovf = %0b, expected %0b", k, ovf, ovf_mem[k][0]);
                end
            end
        end
        $display("tb_yee1d %s: %0d steps, %0d words checked, %0d mismatches, final ovf %0b -> %s",
                 dir, T, checked, errors, ovf, (errors == 0) ? "PASS" : "FAIL");
        if (errors != 0) $fatal(1, "bit mismatch against the software reference");
        $finish;
    end
endmodule
