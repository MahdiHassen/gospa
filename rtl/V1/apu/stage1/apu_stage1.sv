// =============================================================================
// apu_stage1.sv -- APU Stage 1 (ID Generation) Top Level
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Ties the Stage-1 datapath together. Front-end CSR decoding lives upstream
// of this module now (the APU top instantiates act_sram_scanner which emits
// (val, x, y) directly from the activation SRAM). This module accepts that
// stream and produces the FIFO-A bank:
//
//       (val, x, y) handshake (in_valid/in_ready)
//                       |
//                       v
//                  zero_act --(val,x,y)--> position_encode
//                                                |
//                                    (val, Px,Py,Cx,Cy)
//                                                v
//                                         idgen (G x G units)
//                                                |
//                       up to G^2 parallel (valid, a_xy, cid, pid)
//                                                |
//                                    route each to FIFO-A[pid]
//
// FIFO-A[k] stores { activation_value, CID }; PID is IMPLICIT (= the slot
// index k), so it is not stored. Stage 2 (router) drains these read ports.
//
// --- Fan-out / backpressure (the only non-trivial glue here) -----------------
// idgen is a purely combinational fan-out with NO ready signal: for one
// activation it presents up to G^2 results simultaneously, each tagged with a
// PID. For a given activation the G^2 lanes always produce DISTINCT PIDs (the
// (m,n)->(i,j)->PID map is injective over the valid range), so at most one lane
// targets any FIFO-A slot in a cycle -- there is never a write collision on a
// single FIFO-A.
//
// We still need an all-or-nothing join: the activation is consumed (and every
// targeted FIFO-A written) only on a cycle where ALL targeted FIFO-As can
// accept. If any targeted FIFO-A is full we stall the whole front end and
// commit no writes, so no FIFO-A is written twice while we wait on another.
// `s1_ready` carries that stall back up through position_encode -> zero_act ->
// csr_decode via the valid/ready handshake.
//
// NOTE: idgen is instantiated with PIPE = 0 (combinational outputs) so the
// fan-out result is valid in the same cycle as the input coordinates and the
// single-cycle join above is correct. PIPE = 1 would delay the results by a
// cycle and require a skid stage -- out of scope for this integration.
// =============================================================================

`default_nettype none

module apu_stage1 #(
    parameter int H      = 32,   // activation map size (H x H, square)
    parameter int F      = 3,    // weight kernel size (F x F)
    parameter int S      = 1,    // convolution stride
    parameter int DATA_W = 16,   // activation value width in bits
    parameter int FIFO_D = 64,   // FIFO-A depth (power of 2)

    // -- Derived widths (mirror idgen.sv so all ports line up) ----------------
    localparam int E        = (H - F)/S + 1,                  // output map size
    localparam int G        = (F + S - 1)/S,                  // ceil(F/S), integer math
    localparam int NUM_UNIT = G*G,                            // G^2 IDGen lanes
    localparam int N_PID    = F*F,                            // # FIFO-A slots
    localparam int IDX_W    = (H   < 2) ? 1 : $clog2(H),      // coord width
    localparam int CID_W    = (E*E < 2) ? 1 : $clog2(E*E),    // CID width
    localparam int PID_W    = (F*F < 2) ? 1 : $clog2(F*F),    // PID width
    localparam int FIFOA_W  = DATA_W + CID_W,                 // FIFO-A payload width
    localparam int CNT_W    = $clog2(FIFO_D) + 1              // FIFO occupancy width
)(
    input  wire  logic                              clk,
    input  wire  logic                              rst_n,

    // -- (val, x, y) handshake input (from act_sram_scanner upstream) --------
    input  wire  logic                              in_valid,
    input  wire  logic [DATA_W-1:0]                 in_value,
    input  wire  logic [IDX_W-1:0]                  in_x,
    input  wire  logic [IDX_W-1:0]                  in_y,
    output logic                                    in_ready,

    // -- FIFO-A read ports (one per PID slot; consumed by Stage 2) -------------
    output logic [N_PID-1:0]                        fifoa_rd_valid,
    output logic [N_PID-1:0][FIFOA_W-1:0]           fifoa_rd_data,   // {a_xy, cid}
    input  wire  logic [N_PID-1:0]                  fifoa_rd_ready,
    // Per-slot occupancy so Stage 2 can derive almost_empty (count == 1).
    output logic [N_PID-1:0][CNT_W-1:0]             fifoa_count
);

    // -------------------------------------------------------------------------
    // Inter-stage nets (valid/ready handshake on each hop)
    // -------------------------------------------------------------------------
    // upstream handshake feeds zero_act directly (csr_decode lives outside).

    // zero_act -> position_encode
    logic                zp_valid, zp_ready;
    logic [DATA_W-1:0]   zp_value;
    logic [IDX_W-1:0]    zp_x, zp_y;

    // position_encode -> idgen
    logic                pi_valid, pi_ready;
    logic [DATA_W-1:0]   pi_value;
    logic [IDX_W-1:0]    pi_Px, pi_Py, pi_Cx, pi_Cy;

    // idgen fan-out outputs (G^2 lanes, packed)
    logic [NUM_UNIT-1:0]               ig_valid;
    logic [NUM_UNIT-1:0][DATA_W-1:0]   ig_axy;
    logic [NUM_UNIT-1:0][CID_W-1:0]    ig_cid;
    logic [NUM_UNIT-1:0][PID_W-1:0]    ig_pid;

    // FIFO-A bank write side + routing
    logic [N_PID-1:0]               fifoa_wr_valid;
    logic [N_PID-1:0][FIFOA_W-1:0]  fifoa_wr_data;
    logic [N_PID-1:0]               fifoa_wr_ready;
    logic [N_PID-1:0]               need;        // a write is requested to slot p
    logic                           s1_ready;    // all targeted FIFO-As can accept

    // -------------------------------------------------------------------------
    // Zero-activation filter : drop a_xy == 0 before IDGen
    // -------------------------------------------------------------------------
    zero_act #(.H(H), .DATA_W(DATA_W)) u_zero_act (
        .in_valid (in_valid),
        .in_value (in_value),
        .in_x     (in_x),
        .in_y     (in_y),
        .in_ready (in_ready),
        .out_valid(zp_valid),
        .out_value(zp_value),
        .out_x    (zp_x),
        .out_y    (zp_y),
        .out_ready(zp_ready)
    );

    // -------------------------------------------------------------------------
    // Position encode : (x, y) -> (Px, Py, Cx, Cy)
    // -------------------------------------------------------------------------
    position_encode #(.H(H), .S(S), .DATA_W(DATA_W)) u_position_encode (
        .in_valid (zp_valid),
        .in_value (zp_value),
        .x        (zp_x),
        .y        (zp_y),
        .in_ready (zp_ready),
        .out_valid(pi_valid),
        .out_value(pi_value),
        .Px       (pi_Px),
        .Py       (pi_Py),
        .Cx       (pi_Cx),
        .Cy       (pi_Cy),
        .out_ready(pi_ready)
    );

    // position_encode is the last handshake hop; idgen has no ready, so the
    // back-pressure from the FIFO-A bank terminates here.
    assign pi_ready = s1_ready;

    // -------------------------------------------------------------------------
    // IDGen array : (Px,Py,Cx,Cy) -> up to G^2 (valid, a_xy, cid, pid)
    // PIPE = 0 keeps the fan-out combinational (see header note).
    // -------------------------------------------------------------------------
    idgen #(
        .H(H), .F(F), .S(S), .ACT_WIDTH(DATA_W), .PIPE(1'b0)
    ) u_idgen (
        .clk     (clk),
        .rst_n   (rst_n),
        .in_valid(pi_valid),
        .a_xy_in (pi_value),
        .Px      (pi_Px),
        .Py      (pi_Py),
        .Cx      (pi_Cx),
        .Cy      (pi_Cy),
        .valid   (ig_valid),
        .a_xy_out(ig_axy),
        .cid     (ig_cid),
        .pid     (ig_pid)
    );

    // -------------------------------------------------------------------------
    // Route the G^2 lanes to the F^2 FIFO-A slots by PID, with an
    // all-or-nothing join so we never write a slot twice while stalled.
    // -------------------------------------------------------------------------
    always_comb begin
        need = '0;
        // For each FIFO-A slot, find the (unique) lane whose PID targets it.
        for (int p = 0; p < N_PID; p++) begin
            fifoa_wr_data[p] = '0;
            for (int k = 0; k < NUM_UNIT; k++) begin
                if (pi_valid && ig_valid[k] && (ig_pid[k] == PID_W'(p))) begin
                    need[p]          = 1'b1;
                    fifoa_wr_data[p] = {ig_axy[k], ig_cid[k]};  // PID implicit = p
                end
            end
        end
    end

    // All-or-nothing: ready only when every requested FIFO-A can accept.
    always_comb begin
        s1_ready = 1'b1;
        for (int p = 0; p < N_PID; p++)
            if (need[p] && !fifoa_wr_ready[p]) s1_ready = 1'b0;
    end

    // Commit writes only on a fully-accepting cycle.
    always_comb begin
        for (int p = 0; p < N_PID; p++)
            fifoa_wr_valid[p] = need[p] && s1_ready;
    end

    // -------------------------------------------------------------------------
    // FIFO-A bank : one synchronous FIFO per PID slot
    // -------------------------------------------------------------------------
    genvar p;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (p = 0; p < N_PID; p++) begin : g_fifoa
            fifo #(.DATA_WIDTH(FIFOA_W), .DEPTH(FIFO_D)) u_fifoa (
                .clk     (clk),
                .rst_n   (rst_n),
                .wr_valid(fifoa_wr_valid[p]),
                .wr_data (fifoa_wr_data[p]),
                .wr_ready(fifoa_wr_ready[p]),
                .rd_valid(fifoa_rd_valid[p]),
                .rd_data (fifoa_rd_data[p]),
                .rd_ready(fifoa_rd_ready[p]),
                .full    (),
                .empty   (),
                .count   (fifoa_count[p])
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

endmodule

`default_nettype wire
