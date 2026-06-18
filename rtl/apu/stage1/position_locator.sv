module position_locator #(
    parameter int H     = 8,                 // input feature dim (HxH)
    parameter int S     = 1,                 // stride
    parameter int X_WIDTH   = $clog2(H),     // width of a raw coordinate
    parameter int IDX_WIDTH = $clog2(H) + 1  // width for Px/Py/Cx/Cy
)(
    input  logic [X_WIDTH-1:0]       x,
    input  logic [X_WIDTH-1:0]       y,
    output logic [IDX_WIDTH-1:0]     Px,      // x % S
    output logic [IDX_WIDTH-1:0]     Py,      // y % S
    output logic [IDX_WIDTH-1:0]     Cx,      // x / S  (always >= 0)
    output logic [IDX_WIDTH-1:0]     Cy       // y / S  (always >= 0)
);
    always_comb begin
        Px = IDX_WIDTH'(x % S);
        Py = IDX_WIDTH'(y % S);
        Cx = IDX_WIDTH'(x / S);
        Cy = IDX_WIDTH'(y / S);
    end
endmodule