// ---------------------------------------------------------------------
// idgen.sv
// IDGen Bundle (First Module) + IDGen_unit (Second Module).
// ---------------------------------------------------------------------

module idgen #(
    parameter  int H         = 8,      // Input activation map is HxH
    parameter  int F         = 3,      // Weight kernel is FxF
    parameter  int S         = 1,      // Convolution stride
    parameter  int ACT_WIDTH = 8,      // Bit width of the activation value a_xy carried through each unit
    parameter  bit PIPE      = 1'b0,   // 0 = combinational outputs; 1 = register valid/cid/pid/a_xy one cycle

    localparam int E         = (H - F)/S + 1,
    localparam int G         = (F + S - 1)/S,                   // G=ceil(F/S)
    localparam int IDX_WIDTH = (H   < 2) ? 1 : $clog2(H),       // Coord width Px,Py,Cx,Cy
    localparam int CID_WIDTH = (E*E < 2) ? 1 : $clog2(E*E),     // CID field width
    localparam int PID_WIDTH = (F*F < 2) ? 1 : $clog2(F*F),     // PID field width
    localparam int NUM_UNIT  = G*G                              // One IDGen unit per (m,n) in the GxG grid
)(
    input  logic   clk,
    input  logic   rst_n,
    input  logic   in_valid,
    input  logic   [ACT_WIDTH-1:0]                 a_xy_in,
    input  logic   [IDX_WIDTH-1:0]                 Px,
    input  logic   [IDX_WIDTH-1:0]                 Py,
    input  logic   [IDX_WIDTH-1:0]                 Cx,
    input  logic   [IDX_WIDTH-1:0]                 Cy,
    output logic   [NUM_UNIT-1:0 ]                 valid,
    output logic   [NUM_UNIT-1:0 ][ACT_WIDTH-1:0]  a_xy_out,  
    output logic   [NUM_UNIT-1:0 ][CID_WIDTH-1:0]  cid,
    output logic   [NUM_UNIT-1:0 ][PID_WIDTH-1:0]  pid
);

    genvar m, n;
    generate
        for (m = 0; m < G; m++) begin : g_m
            for (n = 0; n < G; n++) begin : g_n
                localparam int IDX = m*G + n;
                idgen_unit #(
                    .M         (m),
                    .N         (n),
                    .F         (F),
                    .S         (S),
                    .H         (H),
                    .E         (E),
                    .ACT_WIDTH (ACT_WIDTH),
                    .PIPE      (PIPE)
                ) u_unit (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .in_valid (in_valid),
                    .a_xy_in  (a_xy_in),
                    .Px       (Px),
                    .Py       (Py),
                    .Cx       (Cx),
                    .Cy       (Cy),
                    .valid    (valid[IDX]),
                    .a_xy_out (a_xy_out[IDX]),
                    .cid      (cid[IDX]),
                    .pid      (pid[IDX])
                );
            end
        end
    endgenerate
endmodule


module idgen_unit #(
    parameter  int M         = 0,      // This unit's group index m
    parameter  int N         = 0,      // This unit's group index n
    parameter  int F         = 3,      // Weight kernel size FxF
    parameter  int S         = 1,      // Stride
    parameter  int H         = 8,      // Input activation map size HxH
    parameter  int E         = (H-F)/S + 1, 
    parameter  int ACT_WIDTH = 8,      // Width of the activation value a_xy passed through
    parameter  bit PIPE      = 1'b0,   // 0 = combinational; 1 = register outputs (valid/cid/pid/a_xy) by one clock

    localparam int IDX_WIDTH = (H   < 2) ? 1 : $clog2(H),   // Coord width Px,Py,Cx,Cy (>=1)
    localparam int CID_WIDTH = (E*E < 2) ? 1 : $clog2(E*E), // CID output width (>=1)
    localparam int PID_WIDTH = (F*F < 2) ? 1 : $clog2(F*F)  // PID output width (>=1)
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,
    input  logic    [ACT_WIDTH-1:0] a_xy_in,
    input  logic    [IDX_WIDTH-1:0] Px,
    input  logic    [IDX_WIDTH-1:0] Py,
    input  logic    [IDX_WIDTH-1:0] Cx,
    input  logic    [IDX_WIDTH-1:0] Cy,
    output logic                    valid,
    output logic    [ACT_WIDTH-1:0] a_xy_out,  
    output logic    [CID_WIDTH-1:0] cid,
    output logic    [PID_WIDTH-1:0] pid
);
    localparam int WIDTH = IDX_WIDTH + 2;

    // Width-matched constants so the index math stays at a known width.
    // Done to keep Verilator lint clean (no WIDTHEXPAND/WIDTHTRUNC).
    localparam logic signed [WIDTH-1:0]   E_W  = WIDTH'(E);       
    localparam logic        [WIDTH-1:0]   F_W  = WIDTH'(F);       
    localparam logic signed [2*WIDTH-1:0] E_2W = (2*WIDTH)'(E);   
    localparam logic signed [2*WIDTH-1:0] F_2W = (2*WIDTH)'(F);   

    logic signed [WIDTH-1:0] a, b;
    logic        [WIDTH-1:0] c, d;
    assign a = $signed(WIDTH'(Cx)) - WIDTH'(M);
    assign b = $signed(WIDTH'(Cy)) - WIDTH'(N);
    assign c = WIDTH'(Px) + WIDTH'(M*S);
    assign d = WIDTH'(Py) + WIDTH'(N*S);

    logic comp1, comp2, comp3, comp4;
    assign comp1 = (0 <= a) && (a < E_W);
    assign comp2 = (0 <= b) && (b < E_W);
    assign comp3 = (c < F_W);   // no lower bound: c,d are unsigned (always >= 0)
    assign comp4 = (d < F_W);

    logic valid_c;
    assign valid_c = in_valid & comp1 & comp2 & comp3 & comp4;

    logic signed [2*WIDTH-1:0] cid_full, pid_full;
    assign cid_full = (2*WIDTH)'(a)*E_2W + (2*WIDTH)'(b);
    assign pid_full = (2*WIDTH)'(c)*F_2W + (2*WIDTH)'(d);

    logic [CID_WIDTH-1:0] cid_c;
    logic [PID_WIDTH-1:0] pid_c;
    assign cid_c = cid_full[CID_WIDTH-1:0];
    assign pid_c = pid_full[PID_WIDTH-1:0];

    generate
        if (PIPE) begin : g_pipe
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid <= 1'b0; cid <= '0; pid <= '0; a_xy_out <= '0;
                end else begin
                    valid <= valid_c; cid <= cid_c; pid <= pid_c; a_xy_out <= a_xy_in;
                end
            end
        end else begin : g_comb
            assign valid    = valid_c;
            assign cid      = cid_c;
            assign pid      = pid_c;
            assign a_xy_out = a_xy_in;
        end
    endgenerate
endmodule



