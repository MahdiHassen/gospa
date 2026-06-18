// =============================================================================
// position_encode.sv -- Activation Position Encoder
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Combinational pass-through stage that turns an activation's absolute
// coordinate (x, y) into the stride-decomposed coordinates the IDGen array
// consumes:
//
//   Px = x % S    -- phase of x within its stride group
//   Py = y % S    -- phase of y within its stride group
//   Cx = x / S    -- coarse x position (stride group index)
//   Cy = y / S    -- coarse y position (stride group index)
//
// idgen.sv takes Px/Py/Cx/Cy as INPUTS and does not compute them, so this
// module sits between zero_act and the IDGen array:
//
//   csr_decode -> zero_act -> [position_encode] -> idgen_array -> FIFO-A
//
// S is a compile-time parameter, so `% S` / `/ S` synthesize to constant
// logic: for the common power-of-two strides (S = 1, 2, 4) they reduce to a
// bit-slice and a shift. For S = 1 this stage is trivial (Px=Py=0, Cx=x, Cy=y).
//
// Backpressure passes straight through (in_ready = out_ready); zero added
// latency, matching zero_act.sv.
// =============================================================================

module position_encode #(
    parameter int H      = 32,   // activation map size (H x H, square)
    parameter int S      = 1,    // convolution stride
    parameter int DATA_W = 16,   // activation value width in bits

    // Coordinate width: matches idgen.sv's IDX_WIDTH so the ports line up.
    localparam int IDX_W = (H < 2) ? 1 : $clog2(H)
)(
    // -- From zero_act --------------------------------------------------------
    input  logic                 in_valid,
    input  logic [DATA_W-1:0]    in_value,
    input  logic [IDX_W-1:0]     in_x,
    input  logic [IDX_W-1:0]     in_y,
    output logic                 in_ready,

    // -- To IDGen array -------------------------------------------------------
    output logic                 out_valid,
    output logic [DATA_W-1:0]    out_value,
    output logic [IDX_W-1:0]     out_Px,
    output logic [IDX_W-1:0]     out_Py,
    output logic [IDX_W-1:0]     out_Cx,
    output logic [IDX_W-1:0]     out_Cy,
    input  logic                 out_ready
);

    // Stride decomposition (constant S -> constant divider / shifter).
    // S is cast to the coordinate width so the operands match (S << 2^IDX_W
    // always, since the stride is far smaller than the map dimension).
    localparam logic [IDX_W-1:0] S_W = IDX_W'(S);
    assign out_Px = in_x % S_W;
    assign out_Py = in_y % S_W;
    assign out_Cx = in_x / S_W;
    assign out_Cy = in_y / S_W;

    // Value and valid pass through unchanged.
    assign out_value = in_value;
    assign out_valid = in_valid;

    // Backpressure passes straight through.
    assign in_ready  = out_ready;

endmodule
