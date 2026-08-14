module position_encode #(
    parameter int H     = 8,                 // input feature dim (HxH)
    parameter int S     = 1,                 // stride
    parameter int DATA_W    = 16,            // activation value width
    parameter int X_WIDTH   = $clog2(H),     // width of a raw coordinate
    parameter int IDX_WIDTH = $clog2(H) + 1  // width for Px/Py/Cx/Cy
)(
    input  logic                     in_valid,
    input  logic [DATA_W-1:0]        in_value,
    input  logic [X_WIDTH-1:0]       x,
    input  logic [X_WIDTH-1:0]       y,
    output logic                     in_ready,

    output logic                     out_valid,
    output logic [DATA_W-1:0]        out_value,
    output logic [IDX_WIDTH-1:0]     Px,      // x % S
    output logic [IDX_WIDTH-1:0]     Py,      // y % S
    output logic [IDX_WIDTH-1:0]     Cx,      // x / S  (always >= 0)
    output logic [IDX_WIDTH-1:0]     Cy,      // y / S  (always >= 0)
    input  logic                     out_ready
);
    always_comb begin
        Px = IDX_WIDTH'(x % S);
        Py = IDX_WIDTH'(y % S);
        Cx = IDX_WIDTH'(x / S);
        Cy = IDX_WIDTH'(y / S);

        out_value = in_value;
        out_valid = in_valid;
        in_ready  = out_ready;
    end
endmodule
