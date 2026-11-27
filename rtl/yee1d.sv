// Fixed-point 1D Yee stencil (docs/rtl.md). Bit-exact with src/fixed1d.rs.
//
// One time step after `start`: cycle 1 updates every H cell, cycle 2 updates
// every interior E cell, and `done` is high for one cycle. All cells update in
// parallel (N multipliers per phase); the design targets clarity, not area.
//
//   H[i] <- sat(H[i] + rnd(CH * (E[i+1] - E[i])))            0 <= i <= N-2
//   E[i] <- sat(E[i] + rnd(CB[i] * (H[i] - H[i-1])) + s[i])   1 <= i <= N-2
//
// E, H, src: W = 18 bits, F = 15 fraction bits. CH, CB: CW = 18 bits,
// CF = 16 fraction bits. rnd(p) = (p + 2^(CF-1)) >>> CF (round half up).
// sat() clamps a 22-bit sum to [-2^(W-1), 2^(W-1) - 1] and sets the sticky
// overflow flag `ovf`. E[0] and E[N-1] stay 0 (PEC ends). `rst` clears the
// fields and `ovf`.

`timescale 1ns / 1ps

module yee1d #(
    parameter int N   = 64,
    parameter int W   = 18,
    parameter int CW  = 18,
    parameter int CF  = 16,
    parameter int SRC = 8
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 start,
    input  logic signed [W-1:0]  src,
    input  logic signed [CW-1:0] ch,
    input  logic signed [CW-1:0] cb [N],
    output logic signed [W-1:0]  e [N],
    output logic signed [W-1:0]  h [N],
    output logic                 done,
    output logic                 ovf
);
    localparam int DW = W + 1;            // difference: exact
    localparam int PW = CW + DW;          // product
    localparam int SW = W + 4;            // sum before saturation
    localparam logic signed [SW-1:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam logic signed [SW-1:0] MINV = -(1 <<< (W - 1));
    // Signed, so that p + HALF stays signed and >>> is an arithmetic shift.
    localparam logic signed [PW-1:0] HALF = 1 <<< (CF - 1);

    typedef enum logic [1:0] {IDLE, PH_H, PH_E} state_t;
    state_t state;

    // The upper bits of q are sign copies (|q| < 2^20), so the truncation to SW bits is exact.
    /* verilator lint_off UNUSEDSIGNAL */
    function automatic logic signed [SW-1:0] rnd(input logic signed [PW-1:0] p);
        logic signed [PW-1:0] q;
        q = (p + HALF) >>> CF;
        return SW'(q);
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    function automatic logic signed [W-1:0] clamp(input logic signed [SW-1:0] v, output logic hit);
        hit = (v > MAXV) || (v < MINV);
        if (v > MAXV) return W'(MAXV);
        if (v < MINV) return W'(MINV);
        return W'(v);
    endfunction

    logic signed [W-1:0] h_next [N];
    logic signed [W-1:0] e_next [N];
    logic [N-1:0] h_hit, e_hit;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            h_next[i] = h[i];
            h_hit[i]  = 1'b0;
            e_next[i] = e[i];
            e_hit[i]  = 1'b0;
        end
        for (int i = 0; i < N - 1; i++) begin
            logic signed [DW-1:0] d;
            logic signed [PW-1:0] p;
            d = DW'(e[i+1]) - DW'(e[i]);
            p = PW'(ch) * PW'(d);
            h_next[i] = clamp(SW'(h[i]) + rnd(p), h_hit[i]);
        end
        for (int i = 1; i < N - 1; i++) begin
            logic signed [DW-1:0] d;
            logic signed [PW-1:0] p;
            logic signed [SW-1:0] s;
            d = DW'(h[i]) - DW'(h[i-1]);
            p = PW'(cb[i]) * PW'(d);
            s = SW'(e[i]) + rnd(p) + ((i == SRC) ? SW'(src) : SW'(0));
            e_next[i] = clamp(s, e_hit[i]);
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done  <= 1'b0;
            ovf   <= 1'b0;
            for (int i = 0; i < N; i++) begin
                e[i] <= '0;
                h[i] <= '0;
            end
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) state <= PH_H;
                PH_H: begin
                    for (int i = 0; i < N; i++) h[i] <= h_next[i];
                    if (|h_hit) ovf <= 1'b1;
                    state <= PH_E;
                end
                PH_E: begin
                    for (int i = 0; i < N; i++) e[i] <= e_next[i];
                    if (|e_hit) ovf <= 1'b1;
                    done  <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
