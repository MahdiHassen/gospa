// =============================================================================
// mac_pipe.sv -- Deeply-pipelined signed multiply-add  r = a*b + c (structural)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Fused multiply-add on mult_pipe (no `*`/`+` operators, no functions). The
// addend `c` rides mult_pipe's aux passthrough so it stays aligned with the
// product at ANY multiplier depth (no manual delay matching); one extra
// registered structural-add stage forms the sum.
//
//   LATENCY = mult_pipe.LATENCY + 1  cycles,  throughput = 1 / cycle.
//
// Computes a *combinational* addend (a*b + c), not a self-accumulating loop --
// feeding a running accumulator back as `c` is the datapath's job.
// Assumes B_W >= 2. Sync reset.
// =============================================================================

`default_nettype none

module mac_pipe #(
    parameter  int A_W = 16,
    parameter  int B_W = 16,
    parameter  int C_W = 32,

    localparam int P_W   = A_W + B_W,
    localparam int OUT_W = ((P_W > C_W) ? P_W : C_W) + 1
)(
    input  wire logic                  clk,
    input  wire logic                  rst_n,
    input  wire logic                  in_valid,
    input  wire logic signed [A_W-1:0] a,
    input  wire logic signed [B_W-1:0] b,
    input  wire logic signed [C_W-1:0] c,
    output logic                       out_valid,
    output logic signed [OUT_W-1:0]    r
);

    // ---- a*b via the deep multiplier; c carried on its aux passthrough -----
    logic                  mv;
    logic signed [P_W-1:0] prod;
    logic        [C_W-1:0] c_al;   // c aligned to prod

    /* verilator lint_off PINCONNECTEMPTY */    // busy unused in the FMA wrapper
    mult_pipe #(.A_W(A_W), .B_W(B_W), .AUX_W(C_W)) u_mult (
        .clk (clk), .rst_n (rst_n), .in_valid (in_valid),
        .a (a), .b (b), .aux_in (c),
        .out_valid (mv), .p (prod), .aux_out (c_al), .busy ()
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // ---- final registered structural add:  r = prod + c_al -----------------
    logic signed [OUT_W-1:0] prod_ext, c_ext;
    assign prod_ext = {{(OUT_W-P_W){prod[P_W-1]}}, prod};
    assign c_ext    = {{(OUT_W-C_W){c_al[C_W-1]}}, c_al};

    logic [OUT_W-1:0] add_s;
    /* verilator lint_off PINCONNECTEMPTY */
    rca_add #(.W(OUT_W)) u_add (
        .x(prod_ext), .y(c_ext), .cin(1'b0), .s(add_s), .cout());
    /* verilator lint_on PINCONNECTEMPTY */

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r         <= '0;
            out_valid <= 1'b0;
        end else begin
            r         <= $signed(add_s);
            out_valid <= mv;
        end
    end

endmodule

`default_nettype wire
