// =============================================================================
// apu_stage2.sv -- APU Stage 2 (PE Assignment) Top Level
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Wraps the routing core + FIFO-B bank:
//
//   FIFO-A read ports (from Stage 1, N_PID lanes)
//        |                |
//        v                v
//   {a_act, a_cid} (PID = slot index, implicit)
//        |
//        v
//      routing  --(b_push[k], b_act[k], b_pid[k], b_cid[k])-->  FIFO-B[k]
//      (start/busy/done framing)                                 (N_PE total)
//
// Backpressure: routing only commits a lane pop on cycles where every selected
// FIFO-B is ready (handled inside routing.sv via b_ready / all_ready). FIFO-B
// is the same parameterized fifo.sv used for FIFO-A; wr_ready falls when full.
//
// `a_almost_empty[p]` is required by routing.sv's lane-advance FSM; we derive
// it here from Stage 1's `fifoa_count[p] == 1`.
// =============================================================================

`default_nettype none

module apu_stage2 #(
    parameter int H      = 32,   // activation map (used to derive E -> CID_W)
    parameter int F      = 3,    // kernel size
    parameter int S      = 1,    // stride
    parameter int N_PE   = 8,    // PE / FIFO-B count
    parameter int DATA_W = 16,   // activation value width
    parameter int FIFO_D = 64,   // FIFO-B depth (power of 2)

    // -- Derived widths (mirror apu_stage1 / routing so ports line up) --------
    localparam int E       = (H - F)/S + 1,
    localparam int N_PID   = F*F,
    localparam int CID_W   = (E*E   < 2) ? 1 : $clog2(E*E),
    localparam int PID_W   = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int FIFOA_W = DATA_W + CID_W,
    localparam int FIFOB_W = DATA_W + PID_W + CID_W,
    localparam int CNT_W   = $clog2(FIFO_D) + 1
)(
    input  wire  logic                              clk,
    input  wire  logic                              rst_n,

    // -- Stage 2 framing ------------------------------------------------------
    input  wire  logic                              start,
    output logic                                    busy,
    output logic                                    done,

    // -- WSP register file: one per PE, MSB-first by PID ---------------------
    //    wsp[k][N_PID-1] = PID 0 .. wsp[k][0] = PID N_PID-1
    input  wire  logic [N_PE-1:0][N_PID-1:0]        wsp,

    // -- FIFO-A read interface (from apu_stage1) -----------------------------
    input  wire  logic [N_PID-1:0]                  fifoa_rd_valid,
    input  wire  logic [N_PID-1:0][FIFOA_W-1:0]     fifoa_rd_data,
    output logic [N_PID-1:0]                        fifoa_rd_ready,
    input  wire  logic [N_PID-1:0][CNT_W-1:0]       fifoa_count,

    // -- FIFO-B read interface (to the PE array) -----------------------------
    output logic [N_PE-1:0]                         fifob_rd_valid,
    output logic [N_PE-1:0][FIFOB_W-1:0]            fifob_rd_data,   // {a_xy, pid, cid}
    input  wire  logic [N_PE-1:0]                   fifob_rd_ready
);

    // -------------------------------------------------------------------------
    // Adapt FIFO-A read ports -> routing's a_* inputs.
    // FIFOA_W layout: {a_xy[MSB..], cid[..LSB]}.
    // -------------------------------------------------------------------------
    logic [N_PID-1:0][DATA_W-1:0]   a_act;
    logic [N_PID-1:0][CID_W-1:0]    a_cid;
    logic [N_PID-1:0]               a_empty;
    logic [N_PID-1:0]               a_almost_empty;
    logic [N_PID-1:0]               a_pop;

    genvar p;
    generate
        for (p = 0; p < N_PID; p++) begin : g_fifoa_unpack
            assign a_act[p]          = fifoa_rd_data[p][FIFOA_W-1 -: DATA_W];
            assign a_cid[p]          = fifoa_rd_data[p][CID_W-1:0];
            assign a_empty[p]        = ~fifoa_rd_valid[p];
            assign a_almost_empty[p] = (fifoa_count[p] == CNT_W'(1));
            assign fifoa_rd_ready[p] = a_pop[p];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Routing core
    // -------------------------------------------------------------------------
    logic [N_PE-1:0]              b_push;
    logic [N_PE-1:0][DATA_W-1:0]  b_act;
    logic [N_PE-1:0][CID_W-1:0]   b_cid;
    logic [N_PE-1:0][PID_W-1:0]   b_pid;
    logic [N_PE-1:0]              fifob_wr_ready;   // = b_ready into routing

    routing #(
        .N_PID(N_PID), .N_PE(N_PE),
        .ACT_WIDTH(DATA_W), .CID_WIDTH(CID_W)
    ) u_routing (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .wsp           (wsp),
        .a_act         (a_act),
        .a_cid         (a_cid),
        .a_empty       (a_empty),
        .a_almost_empty(a_almost_empty),
        .a_pop         (a_pop),
        .b_ready       (fifob_wr_ready),
        .b_push        (b_push),
        .b_act         (b_act),
        .b_cid         (b_cid),
        .b_pid         (b_pid)
    );

    // -------------------------------------------------------------------------
    // FIFO-B bank: one fifo per PE. Payload = {a_xy, pid, cid}.
    // wr_ready feeds routing.b_ready, so a full FIFO-B stalls the head pop.
    // -------------------------------------------------------------------------
    genvar k;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (k = 0; k < N_PE; k++) begin : g_fifob
            fifo #(.DATA_WIDTH(FIFOB_W), .DEPTH(FIFO_D)) u_fifob (
                .clk     (clk),
                .rst_n   (rst_n),
                .wr_valid(b_push[k]),
                .wr_data ({b_act[k], b_pid[k], b_cid[k]}),
                .wr_ready(fifob_wr_ready[k]),
                .rd_valid(fifob_rd_valid[k]),
                .rd_data (fifob_rd_data[k]),
                .rd_ready(fifob_rd_ready[k]),
                .full    (),
                .empty   (),
                .count   ()
            );
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

endmodule

`default_nettype wire
