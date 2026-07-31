// =============================================================================
// apu_stage2.sv -- APU Stage 2 (PE Assignment) Top Level (V2 "act" dataflow)
// =============================================================================
// GoSPA Project -- Team 19, ECE 720 (Spring 2026)
//
// Wraps the routing core + FIFO-B bank:
//
//   FIFO-A read ports (from Stage 1, N_PID lanes, each NUM_MULTS-wide)
//        |                |
//        v                v
//   {a_act[M], a_cid[M]} (PID = lane index, implicit)
//        |
//        v
//      routing  --(one M-wide BEAT: pid, lane_valid[M], act[M], cid[M])-->
//      (start/busy/done framing)           FIFO-B[k]  (N_PE total, 1 beat/entry)
//
// The router pops up to NUM_MULTS same-PID entries per cycle from the current
// lane and packs them into one beat; each FIFO-B entry is one beat. On the read
// side the beat is unpacked into structured ports that line up with pe_array.
//
// Backpressure: routing commits a beat only on cycles where every selected
// FIFO-B is ready (all_ready inside routing.sv via b_ready). FIFO-A is the same
// fifo.sv used elsewhere, here with PORT_WIDTH = NUM_MULTS so the read window
// exposes up to M entries; FIFO-B keeps PORT_WIDTH = 1 (one beat per push/pop).
// =============================================================================

`default_nettype none

module apu_stage2 #(
    parameter int H         = 32,   // activation map (used to derive E -> CID_W)
    parameter int F         = 3,    // kernel size
    parameter int S         = 1,    // stride
    parameter int N_PE      = 8,    // PE / FIFO-B count
    parameter int NUM_MULTS = 4,    // activations per beat = FIFO-A read width
    parameter int DATA_W    = 16,   // activation value width
    parameter int FIFO_D    = 64,   // FIFO depth (power of 2)

    // -- Derived widths (mirror apu_stage1 / routing so ports line up) --------
    localparam int E       = (H - F)/S + 1,
    localparam int N_PID   = F*F,
    localparam int CID_W   = (E*E   < 2) ? 1 : $clog2(E*E),
    localparam int PID_W   = (N_PID < 2) ? 1 : $clog2(N_PID),
    localparam int FIFOA_W = DATA_W + CID_W,
    localparam int CNT_W   = $clog2(FIFO_D) + 1,
    // One FIFO-B entry = one M-wide beat: {pid, lane_valid[M], act[M], cid[M]}.
    localparam int BEAT_W  = PID_W + NUM_MULTS + NUM_MULTS*DATA_W + NUM_MULTS*CID_W
)(
    input  wire  logic                                              clk,
    input  wire  logic                                              rst_n,

    // -- Stage 2 framing ------------------------------------------------------
    input  wire  logic                                             start,
    output logic                                                   busy,
    output logic                                                   done,

    // -- Per-PE WSP from the PE array, LSB-first by PID ----------------------
    //    wsp[k][p] = 1 means PE k has a weight at PID p (pe_mem.wsp_q)
    input  wire  logic [N_PE-1:0][N_PID-1:0]                       wsp,

    // -- FIFO-A read interface (from apu_stage1, M-wide per lane) -------------
    input  wire  logic [N_PID-1:0][NUM_MULTS-1:0]                  fifoa_rd_valid,
    input  wire  logic [N_PID-1:0][NUM_MULTS-1:0][FIFOA_W-1:0]     fifoa_rd_data,
    output logic [N_PID-1:0][NUM_MULTS-1:0]                        fifoa_rd_ready,
    input  wire  logic [N_PID-1:0][CNT_W-1:0]                      fifoa_count,

    // -- FIFO-B read interface (to the PE array, one M-wide beat) -------------
    output logic [N_PE-1:0]                                        fifob_rd_valid,
    output logic [N_PE-1:0][PID_W-1:0]                             fifob_rd_pid,
    output logic [N_PE-1:0][NUM_MULTS-1:0]                         fifob_rd_lane_valid,
    output logic [N_PE-1:0][NUM_MULTS-1:0][DATA_W-1:0]             fifob_rd_act,
    output logic [N_PE-1:0][NUM_MULTS-1:0][CID_W-1:0]              fifob_rd_cid,
    input  wire  logic [N_PE-1:0]                                  fifob_rd_ready
);

    // -------------------------------------------------------------------------
    // Adapt FIFO-A read ports -> routing's a_* inputs.
    // FIFOA_W layout per entry: {a_xy[MSB..], cid[..LSB]}.
    // -------------------------------------------------------------------------
    logic [N_PID-1:0][NUM_MULTS-1:0][DATA_W-1:0] a_act;
    logic [N_PID-1:0][NUM_MULTS-1:0][CID_W-1:0]  a_cid;
    logic [N_PID-1:0][NUM_MULTS-1:0]             a_rvalid;
    logic [N_PID-1:0][NUM_MULTS-1:0]             a_pop;

    genvar p, i;
    generate
        for (p = 0; p < N_PID; p++) begin : g_fifoa_unpack
            for (i = 0; i < NUM_MULTS; i++) begin : g_lane
                assign a_act[p][i]    = fifoa_rd_data[p][i][FIFOA_W-1 -: DATA_W];
                assign a_cid[p][i]    = fifoa_rd_data[p][i][CID_W-1:0];
                assign a_rvalid[p][i] = fifoa_rd_valid[p][i];
            end
            assign fifoa_rd_ready[p] = a_pop[p];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Routing core
    // -------------------------------------------------------------------------
    logic [N_PE-1:0]                              b_push;
    logic [N_PE-1:0][PID_W-1:0]                   b_pid;
    logic [N_PE-1:0][NUM_MULTS-1:0]              b_lane_valid;
    logic [N_PE-1:0][NUM_MULTS-1:0][DATA_W-1:0]  b_act;
    logic [N_PE-1:0][NUM_MULTS-1:0][CID_W-1:0]   b_cid;
    logic [N_PE-1:0]                              fifob_wr_ready;   // = b_ready into routing

    routing #(
        .N_PID(N_PID), .N_PE(N_PE), .NUM_MULTS(NUM_MULTS),
        .ACT_WIDTH(DATA_W), .CID_WIDTH(CID_W), .CNT_WIDTH(CNT_W)
    ) u_routing (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .busy        (busy),
        .done        (done),
        .wsp         (wsp),
        .a_act       (a_act),
        .a_cid       (a_cid),
        .a_rvalid    (a_rvalid),
        .a_count     (fifoa_count),
        .a_pop       (a_pop),
        .b_ready     (fifob_wr_ready),
        .b_push      (b_push),
        .b_pid       (b_pid),
        .b_lane_valid(b_lane_valid),
        .b_act       (b_act),
        .b_cid       (b_cid)
    );

    // -------------------------------------------------------------------------
    // FIFO-B bank: one fifo per PE, one beat per entry.
    // Beat word (MSB->LSB): {pid, lane_valid[M], act[M], cid[M]}.
    // wr_ready feeds routing.b_ready, so a full FIFO-B stalls the beat push.
    // -------------------------------------------------------------------------
    genvar k;
    /* verilator lint_off PINCONNECTEMPTY */
    generate
        for (k = 0; k < N_PE; k++) begin : g_fifob
            logic [BEAT_W-1:0] wr_beat;
            logic [BEAT_W-1:0] rd_beat;

            assign wr_beat = {b_pid[k], b_lane_valid[k], b_act[k], b_cid[k]};

            fifo #(.DATA_WIDTH(BEAT_W), .DEPTH(FIFO_D)) u_fifob (
                .clk     (clk),
                .rst_n   (rst_n),
                .wr_valid(b_push[k]),
                .wr_data (wr_beat),
                .wr_ready(fifob_wr_ready[k]),
                .rd_valid(fifob_rd_valid[k]),
                .rd_data (rd_beat),
                .rd_ready(fifob_rd_ready[k]),
                .full    (),
                .empty   (),
                .count   ()
            );

            // Unpack the beat back into structured read ports (match pe_array).
            assign fifob_rd_cid[k]        = rd_beat[NUM_MULTS*CID_W-1 -: NUM_MULTS*CID_W];
            assign fifob_rd_act[k]        = rd_beat[NUM_MULTS*CID_W + NUM_MULTS*DATA_W-1 -: NUM_MULTS*DATA_W];
            assign fifob_rd_lane_valid[k] = rd_beat[NUM_MULTS*CID_W + NUM_MULTS*DATA_W + NUM_MULTS-1 -: NUM_MULTS];
            assign fifob_rd_pid[k]        = rd_beat[BEAT_W-1 -: PID_W];
        end
    endgenerate
    /* verilator lint_on PINCONNECTEMPTY */

endmodule

`default_nettype wire
