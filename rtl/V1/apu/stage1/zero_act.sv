// =============================================================================
// zero_act.sv -- Zero Activation Filter
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Purely combinational pass-through filter. Suppresses the valid signal for
// any activation whose value is zero, preventing zero activations from ever
// reaching the IDGen array. This avoids G^2 wasted comparisons per zero.
//
// Equivalent to the Python model's: "if val == 0: continue"
//
// Backpressure passes straight through: in_ready = out_ready.
// This module adds zero combinational delay to the ready path.
// =============================================================================

module zero_act #(
    parameter int H      = 32,   // activation map size (for coord widths)
    parameter int DATA_W = 16    // activation value width in bits
)(
    // -- From csr_decode ------------------------------------------------------
    input  logic                   in_valid,
    input  logic [DATA_W-1:0]      in_value,
    input  logic [$clog2(H)-1:0]   in_x,
    input  logic [$clog2(H)-1:0]   in_y,
    output logic                   in_ready,

    // -- To IDGen array -------------------------------------------------------
    output logic                   out_valid,
    output logic [DATA_W-1:0]      out_value,
    output logic [$clog2(H)-1:0]   out_x,
    output logic [$clog2(H)-1:0]   out_y,
    input  logic                   out_ready
);

    // Filter: suppress valid for zero-valued activations
    assign out_valid = in_valid && (in_value != '0);

    // Data passes through unchanged
    assign out_value = in_value;
    assign out_x     = in_x;
    assign out_y     = in_y;

    // Backpressure passes straight through
    assign in_ready  = out_ready;

endmodule
