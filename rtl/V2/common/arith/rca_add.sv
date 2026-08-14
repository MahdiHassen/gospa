// =============================================================================
// rca_add.sv -- Structural ripple-carry adder (combinational, no `+`, no function)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// W-bit full-adder chain:  s = x + y + cin  (with carry-out), built from bitwise
// gates in an always_comb block. Reused by mult_pipe / mac_pipe so the
// arithmetic contains no `+`/`*` operators and no SystemVerilog functions.
// Purely combinational; the pipeline registers live in the parent module.
// The carry is a block-local scalar (not a module signal) so the ripple order
// is unambiguous.
// =============================================================================

`default_nettype none

module rca_add #(
    parameter int W = 32
)(
    input  wire logic [W-1:0] x,
    input  wire logic [W-1:0] y,
    input  wire logic         cin,
    output logic [W-1:0]      s,
    output logic              cout
);

    always_comb begin
        logic ci;
        ci = cin;
        for (int i = 0; i < W; i++) begin
            s[i] = x[i] ^ y[i] ^ ci;
            ci   = (x[i] & y[i]) | (ci & (x[i] ^ y[i]));
        end
        cout = ci;
    end

endmodule

`default_nettype wire
